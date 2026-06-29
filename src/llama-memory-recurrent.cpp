#include "llama-memory-recurrent.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_memory_recurrent
//

llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s,
                     bool   offload,
                 uint32_t   mem_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter) : hparams(model.hparams), n_seq_max(n_seq_max) {
    const int32_t n_layer = hparams.n_layer();

    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;
    rs_idx.assign(n_seq_max, 0);

    cells.clear();
    cells.resize(mem_size);

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    r_l.resize(n_layer);
    s_l.resize(n_layer);

    for (int i = 0; i < n_layer; i++) {
        if (filter && !filter(i)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, i, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for rs cache");
        }

        const uint32_t n_rows = mem_size * (1 + n_rs_seq);
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
        r_l[i] = r;
        s_l[i] = s;
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for rs cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_r = size_r_bytes();
        const size_t memory_size_s = size_s_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_r + memory_size_s) / (1024.0f * 1024.0f), mem_size, n_layer, n_seq_max, n_rs_seq,
                ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
                ggml_type_name(type_s), (float)memory_size_s / (1024.0f * 1024.0f));
    }
}

void llama_memory_recurrent::clear(bool data) {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
        cells[i].src0 = -1;
        cells[i].tail = -1;
    }

    head = 0;
    used = 0;

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);

    // Discard any saved recurrent checkpoint — the tensor data
    // was just zeroed, so a stale restore would be wrong.
    clear_checkpoint();
}

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();
    if (rm_all) {
        if (seq_id >= 0) {
            set_rs_idx(seq_id, 0);
        } else {
            std::fill(rs_idx.begin(), rs_idx.end(), 0);
        }
    }

    // models like Mamba or RWKV can't have a state partially erased at the end
    // of the sequence because their state isn't preserved for previous tokens
    if (seq_id >= (int64_t) size) {
        // could be fatal
        return false;
    }
    if (0 <= seq_id) {
        int32_t & tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];

            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                const llama_pos rollback = cell.pos - (p0 - 1);
                if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
                    set_rs_idx(seq_id, (uint32_t) rollback);
                    cell.pos = p0 - 1;
                    return true;
                }
                // Rollback too large — invalidate the cell's tail pointer
                // so callers don't try to reuse a cell with stale state.
                tail_id = -1;
                return false;
            }
            // invalidate tails which will be cleared
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
            }
        }
    } else {
        // seq_id is negative, then the range should include everything or nothing
        if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
            //printf("[DEBUG] inside `llama_memory_recurrent::seq_rm`: `seq_id` is negative, so returning false\n");
            return false;
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].src = -1;
                cells[i].src0 = -1;
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
        // When two sequences share a cell, s_copy() reads rs_idx from the
        // first seq_id in the set.  If rs_idx differed, one sequence's
        // rollback state would be silently ignored.  Assert they match now
        // so we catch any future call site that breaks this invariant.
        if (n_rs_seq > 0) {
            GGML_ASSERT(rs_idx[seq_id_src] == rs_idx[seq_id_dst]);
        }

        auto & tail_src = cells[seq_id_src];
        auto & tail_dst = cells[seq_id_dst];
        if (tail_dst.tail >= 0) {
            // clear destination seq_id if it wasn't empty
            auto & cell_dst = cells[tail_dst.tail];

            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst.tail = -1;
            if (cell_dst.seq_id.empty()) {
                cell_dst.pos = -1;
                cell_dst.src = -1;
                cell_dst.src0 = -1;
                used -= 1;
            }
        }
        if (tail_src.tail >= 0) {
            auto & cell_src = cells[tail_src.tail];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst.tail = tail_src.tail;
        }
    }
}

void llama_memory_recurrent::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        if ((llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
        }

        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_memory_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0 || (size_t) seq_id >= rs_idx.size()) {
        return;
    }
    rs_idx[seq_id] = (idx > n_rs_seq) ? n_rs_seq : idx;
}

int32_t llama_memory_recurrent::get_cell_tail(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= size) {
        return -1;
    }
    return cells[seq_id].tail;
}

void llama_memory_recurrent::cell_zero(llama_seq_id seq_id) {
    if (seq_id < 0 || (uint32_t) seq_id >= size) {
        return;
    }

    int32_t cell_idx = cells[seq_id].tail;
    // Fallback: if tail is invalid (e.g., after resize cleared it),
    // scan for the cell that owns this seq_id.
    if (cell_idx < 0 || (uint32_t) cell_idx >= size) {
        cell_idx = -1;
        for (uint32_t i = 0; i < size; i++) {
            if (cells[i].has_seq_id(seq_id)) {
                cell_idx = (int32_t)i;
                break;
            }
        }
        if (cell_idx < 0) {
            return;
        }
    }

    // Tensor layout is plane-interleaved:
    //   [primary_cell_0, ..., primary_cell_{N-1},
    //    snap_1_cell_0, ..., snap_1_cell_{N-1}, ...]
    // where N = size.  A cell's planes are at row = cell_idx + plane * size.
    // We zero every plane individually to match s_copy() and build_rs().

    const int32_t n_layer = hparams.n_layer();
    std::vector<uint8_t> zero_buf;

    for (int il = 0; il < n_layer; il++) {
        if (r_l[il]) {
            size_t row_bytes = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
            zero_buf.assign(row_bytes, 0);
            // zero primary row
            size_t offset_primary = (size_t)cell_idx * row_bytes;
            ggml_backend_tensor_set(r_l[il], zero_buf.data(), offset_primary, row_bytes);
            // zero each snapshot row
            for (uint32_t k = 1; k <= n_rs_seq; k++) {
                size_t offset_snap = ((size_t)cell_idx + (size_t)k * (size_t)size) * row_bytes;
                ggml_backend_tensor_set(r_l[il], zero_buf.data(), offset_snap, row_bytes);
            }
        }
        if (s_l[il]) {
            size_t row_bytes = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            zero_buf.assign(row_bytes, 0);
            size_t offset_primary = (size_t)cell_idx * row_bytes;
            ggml_backend_tensor_set(s_l[il], zero_buf.data(), offset_primary, row_bytes);
            for (uint32_t k = 1; k <= n_rs_seq; k++) {
                size_t offset_snap = ((size_t)cell_idx + (size_t)k * (size_t)size) * row_bytes;
                ggml_backend_tensor_set(s_l[il], zero_buf.data(), offset_snap, row_bytes);
            }
        }
    }

    // Reset cell metadata to reflect that the state is now zeroed/invalid.
    // This prevents find_slot from seeing stale positions and triggering
    // "non-consecutive" warnings or using incorrect src/src0 references.
    //
    // seq_id MUST be cleared here.  When seq_rm fails a bounded rollback
    // it sets tail = -1 before returning, so find_slot's tail-based cell
    // lookup returns has_cell = false; find_slot then scans for an empty cell
    // via is_empty().  If seq_id survived, is_empty() stays false and the
    // scan wraps past the end of the cells array — assertion failure
    // (empty_cell.is_empty()) or out-of-bounds access when size == 1.
    //
    // Clearing seq_id makes the cell genuinely empty and findable by the
    // empty-cell scan without breaking any other code path:
    //   - resize() tail recovery runs before any cell_zero call;
    //   - cell theft by another seq is not a concern because tail is already
    //     -1 — find_slot will go through the fresh-allocation path and
    //     rebuild ownership from scratch.
    cells[cell_idx].seq_id.clear();
    cells[cell_idx].pos = -1;
    cells[cell_idx].src = -1;
    cells[cell_idx].src0 = -1;

    // Reset rollback index so find_slot uses zero initialization
    set_rs_idx(seq_id, 0);
}

void llama_memory_recurrent::cell_zero_snapshots(uint32_t cell_idx) {
    if (cell_idx >= size) {
        return;
    }
    if (n_rs_seq == 0) {
        return;
    }

    // Tensor layout is plane-interleaved.
    // Zero each snapshot plane individually (plane = 1..n_rs_seq).
    const int32_t n_layer = hparams.n_layer();
    std::vector<uint8_t> zero_buf;

    for (int il = 0; il < n_layer; il++) {
        if (r_l[il]) {
            size_t row_bytes = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
            zero_buf.assign(row_bytes, 0);
            for (uint32_t k = 1; k <= n_rs_seq; k++) {
                size_t offset_snap = ((size_t)cell_idx + (size_t)k * (size_t)size) * row_bytes;
                ggml_backend_tensor_set(r_l[il], zero_buf.data(), offset_snap, row_bytes);
            }
        }
        if (s_l[il]) {
            size_t row_bytes = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            zero_buf.assign(row_bytes, 0);
            for (uint32_t k = 1; k <= n_rs_seq; k++) {
                size_t offset_snap = ((size_t)cell_idx + (size_t)k * (size_t)size) * row_bytes;
                ggml_backend_tensor_set(s_l[il], zero_buf.data(), offset_snap, row_bytes);
            }
        }
    }
}

void llama_memory_recurrent::cell_copy_primary_to_snapshots(uint32_t cell_idx) {
    if (cell_idx >= size) {
        return;
    }
    if (n_rs_seq == 0) {
        return;
    }

    const int32_t n_layer = hparams.n_layer();

    for (int i = 0; i < n_layer; i++) {
        if (r_l[i]) {
            size_t row_bytes = ggml_row_size(r_l[i]->type, hparams.n_embd_r());
            size_t offset_primary = (size_t)cell_idx * row_bytes;
            // Read primary row to CPU buffer, then copy to each snapshot row
            std::vector<uint8_t> buf(row_bytes);
            ggml_backend_tensor_get(r_l[i], buf.data(), offset_primary, row_bytes);
            for (uint32_t k = 1; k <= n_rs_seq; k++) {
                size_t offset_snap = ((size_t)cell_idx + (size_t)k * (size_t)size) * row_bytes;
                ggml_backend_tensor_set(r_l[i], buf.data(), offset_snap, row_bytes);
            }
        }
        if (s_l[i]) {
            size_t row_bytes = ggml_row_size(s_l[i]->type, hparams.n_embd_s());
            size_t offset_primary = (size_t)cell_idx * row_bytes;
            std::vector<uint8_t> buf(row_bytes);
            ggml_backend_tensor_get(s_l[i], buf.data(), offset_primary, row_bytes);
            for (uint32_t k = 1; k <= n_rs_seq; k++) {
                size_t offset_snap = ((size_t)cell_idx + (size_t)k * (size_t)size) * row_bytes;
                ggml_backend_tensor_set(s_l[i], buf.data(), offset_snap, row_bytes);
            }
        }
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

llama_memory_context_ptr llama_memory_recurrent::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                if (n_rs_seq > 0) {
                    // [TAG_RECURRENT_ROLLBACK_SPLITS]
                    // TODO: recurrent state rollback does not support equal splits
                    ubatch = balloc.split_seq(n_ubatch);
                } else {
                    // TODO: non-sequential equal split can be done if using unified KV cache
                    //       for simplicity, we always use sequential equal split for now
                    ubatch = balloc.split_equal(n_ubatch, true);
                }
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        if (!prepare(ubatches)) {
            break;
        }

        return std::make_unique<llama_memory_recurrent_context>(this, std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_recurrent::init_full() {
    return std::make_unique<llama_memory_recurrent_context>(this);
}

llama_memory_context_ptr llama_memory_recurrent::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_recurrent::prepare(const std::vector<llama_ubatch> & ubatches) {
    // simply remember the full state because it is very small for this type of cache
    // TODO: optimize
    auto org_cells = cells;
    auto org_used = used;
    auto org_head = head;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        if (!find_slot(ubatch)) {
            success = false;
            break;
        }
    }

    // restore the original state
    cells = std::move(org_cells);
    used = org_used;
    head = org_head;

    return success;
}

bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*n_seqs) {
        head = 0;
    }

    // For recurrent state architectures (like Mamba or RWKV),
    // each cache cell can store the state for a whole sequence.
    // A slot should be always be contiguous.

    // can only process batches with an equal number of new tokens in each sequence
    GGML_ASSERT(ubatch.equal_seqs());

    int32_t min = size - 1;
    int32_t max = 0;

    // everything should fit if all seq_ids are smaller than the max
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens; // first token of sequence set s
        const uint32_t n_seq_id = ubatch.n_seq_id[i];

        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];

            if (seq_id < 0 || (uint32_t) seq_id >= size) {
                // too big seq_id
                // TODO: would it be possible to resize the cache instead?
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, n_seq_max);
                return false;
            }
            if (j > 0) {
                auto & seq = cells[seq_id];
                if (seq.tail >= 0) {
                    auto & cell = cells[seq.tail];
                    // clear cells from seq_ids that become shared
                    // (should not normally happen, but let's handle it anyway)
                    cell.seq_id.erase(seq_id);
                    seq.tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        used -= 1;
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::vector<int32_t> tails_verif;
        tails_verif.assign(size, -1);
        for (uint32_t i = 0; i < size; ++i) {
            auto & cell = cells[i];
            for (llama_seq_id seq_id : cell.seq_id) {
                if (tails_verif[seq_id] != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                }
                tails_verif[seq_id] = i;
            }
        }
        for (uint32_t i = 0; i < size; ++i) {
            if (tails_verif[i] != cells[i].tail) {
                LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
            }
        }
    }
#endif

    // find next empty cell
    uint32_t next_empty_cell = head;

    for (uint32_t i = 0; i < size; ++i) {
        if (next_empty_cell >= size) { next_empty_cell -= size; }
        auto & cell = cells[next_empty_cell];
        if (cell.is_empty()) { break; }
        next_empty_cell += 1;
    }

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        auto & seq_meta = cells[seq_id];
        bool has_cell = false;
        if (seq_meta.tail >= 0) {
            auto & cell = cells[seq_meta.tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            // does this seq_id "own" the cell?
            if (cell.seq_id.size() == 1) { has_cell = true; }
        }
        if (!has_cell) {
            auto & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            // copy old tail into the empty cell
            if (seq_meta.tail >= 0) {
                auto & orig_cell = cells[seq_meta.tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = orig_cell.src;
                orig_cell.seq_id.erase(seq_id);
                empty_cell.seq_id.insert(seq_id); // will be overwritten
                GGML_ASSERT(!orig_cell.is_empty()); // has at least one remaining seq_id
            }
            seq_meta.tail = next_empty_cell;
            // find next empty cell
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                }
            }
        }
        if (min > seq_meta.tail) { min = seq_meta.tail; }
        if (max < seq_meta.tail) { max = seq_meta.tail; }
    }

    // gather and re-order
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const int32_t dst_id = s + min;
        const int32_t src_id = cells[ubatch.seq_id[i][0]].tail;
        if (dst_id != src_id) {
            auto & dst_cell = cells[dst_id];
            auto & src_cell = cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.seq_id, src_cell.seq_id);

            // swap tails
            for (uint32_t j = 0; j < size; ++j) {
                int32_t & tail = cells[j].tail;
                if (tail == src_id) {
                    tail = dst_id;
                } else if (tail == dst_id) {
                    tail = src_id;
                }
            }
        }
    }

    // update the pos of the used seqs
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];
        const int32_t cell_id = s + min;
        auto & cell = cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                __func__, last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens);
        }
        cell.pos = last_pos;
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            cell.seq_id.insert(seq_id);
            cells[seq_id].tail = cell_id;
        }
    }

    // Find first cell without src refs, to use as the zero-ed state
    {
        // TODO: bake-in src refcounts in the cell metadata
        std::vector<int32_t> refcounts(size, 0);
        for (size_t i = 0; i < size; ++i) {
            const int32_t src = cells[i].src;
            if (src >= 0) {
                refcounts[src] += 1;
            }
        }

        rs_z = -1;
        for (int i = min; i <= max; ++i) {
            if (refcounts[i] == 0) {
                rs_z = i;
                break;
            }
        }

        for (int i = min; i <= max; ++i) {
            if (cells[i].src < 0) {
                GGML_ASSERT(rs_z >= 0);
                cells[i].src0 = rs_z;
                // Fresh cell — zero its snapshot rows to prevent stale data
                // from a previous sequence from being read via rs_idx rollback.
                // build_rs zeroes the rs_z cell's primary row (or, with our fix,
                // all its snapshots too) but when rs_z != this cell (i.e. the
                // cell is fresh and rs_z points elsewhere), only the primary row
                // is cleared through src → src0 → build_rs copy.  The snapshot
                // rows of this freshly allocated cell would otherwise retain
                // whatever data was last written there.
                cell_zero_snapshots(i);
            } else {
                // Stage the source ids for all used cells to allow correct seq_* behavior
                // and still make these values available when setting the inputs
                cells[i].src0 = cells[i].src;
            }
            cells[i].src = i; // avoid moving or clearing twice
        }
    }

    // allow getting the range of used cells, from head to head + n
    head = min;
    n    = max - min + 1;
    used = std::count_if(cells.begin(), cells.end(),
        [](const mem_cell & cell){ return !cell.is_empty(); });

    // sanity check
    return n >= n_seqs;
}

bool llama_memory_recurrent::get_can_shift() const {
    // shifting the pos is trivial for recurrent models
    return true;
}

//
// Recurrent state checkpoint — save/restore around shrink/expand
//

void llama_memory_recurrent::save_checkpoint() {
    // Only save when there is something to preserve.
    if (cells.empty()) {
        return;
    }

    const int32_t n_layer = hparams.n_layer();
    const uint32_t rows_per_cell = 1 + n_rs_seq;

    // Determine per-layer row sizes (in bytes).
    std::vector<size_t> r_row_bytes(n_layer, 0);
    std::vector<size_t> s_row_bytes(n_layer, 0);
    for (int i = 0; i < n_layer; i++) {
        if (r_l[i]) {
            r_row_bytes[i] = ggml_row_size(r_l[i]->type, hparams.n_embd_r());
        }
        if (s_l[i]) {
            s_row_bytes[i] = ggml_row_size(s_l[i]->type, hparams.n_embd_s());
        }
    }

    // Count active cells and estimate total buffer size.
    size_t total_r = 0;
    size_t total_s = 0;
    uint32_t active_count = 0;
    for (uint32_t i = 0; i < size; ++i) {
        if (!cells[i].is_empty()) {
            ++active_count;
            for (int l = 0; l < n_layer; ++l) {
                total_r += r_row_bytes[l] * rows_per_cell;
                total_s += s_row_bytes[l] * rows_per_cell;
            }
        }
    }

    if (active_count == 0) {
        return;
    }

    recr_checkpoint_cells.clear();
    recr_checkpoint_cells.reserve(active_count);
    recr_checkpoint_rows_per_cell = rows_per_cell;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].is_empty()) {
            continue;
        }

        recr_checkpoint_cell cell;
        cell.cell_idx = i;
        cell.pos = cells[i].pos;
        cell.src = cells[i].src;
        cell.src0 = cells[i].src0;
        cell.tail = cells[i].tail;
        cell.seq_id = cells[i].seq_id;

        size_t cell_r_size = 0;
        size_t cell_s_size = 0;
        for (int l = 0; l < n_layer; ++l) {
            cell_r_size += r_row_bytes[l] * rows_per_cell;
            cell_s_size += s_row_bytes[l] * rows_per_cell;
        }

        cell.r_data.resize(cell_r_size);
        cell.s_data.resize(cell_s_size);

        // Read each plane individually (plane-interleaved layout).
        // Primary plane: row = cell_idx, then snapshot planes: row = cell_idx + p * size.
        size_t r_off = 0;
        size_t s_off = 0;
        for (int l = 0; l < n_layer; ++l) {
            if (r_l[l]) {
                for (uint32_t p = 0; p < rows_per_cell; p++) {
                    size_t offset = ((size_t)i + (size_t)p * (size_t)size) * r_row_bytes[l];
                    ggml_backend_tensor_get(r_l[l], cell.r_data.data() + r_off, offset, r_row_bytes[l]);
                    r_off += r_row_bytes[l];
                }
            }
            if (s_l[l]) {
                for (uint32_t p = 0; p < rows_per_cell; p++) {
                    size_t offset = ((size_t)i + (size_t)p * (size_t)size) * s_row_bytes[l];
                    ggml_backend_tensor_get(s_l[l], cell.s_data.data() + s_off, offset, s_row_bytes[l]);
                    s_off += s_row_bytes[l];
                }
            }
        }

        recr_checkpoint_cells.push_back(std::move(cell));
    }
}

void llama_memory_recurrent::restore_checkpoint() {
    if (recr_checkpoint_cells.empty()) {
        return;
    }

    const int32_t n_layer = hparams.n_layer();

    // Determine current per-layer row sizes (may differ if n_rs_seq changed,
    // but in the shrink→expand round-trip they should match).
    std::vector<size_t> r_row_bytes(n_layer, 0);
    std::vector<size_t> s_row_bytes(n_layer, 0);
    for (int i = 0; i < n_layer; i++) {
        if (r_l[i]) {
            r_row_bytes[i] = ggml_row_size(r_l[i]->type, hparams.n_embd_r());
        }
        if (s_l[i]) {
            s_row_bytes[i] = ggml_row_size(s_l[i]->type, hparams.n_embd_s());
        }
    }

    for (auto & cell : recr_checkpoint_cells) {
        // Guard: skip if the cell index is out of range after expand.
        if (cell.cell_idx >= size) {
            continue;
        }

        // Restore cell metadata.
        cells[cell.cell_idx].pos = cell.pos;
        cells[cell.cell_idx].src = cell.src;
        cells[cell.cell_idx].src0 = cell.src0;
        // Guard: if tail points to a cell that was truncated during shrink,
        // clamp to -1 to prevent dangling pointer access.
        cells[cell.cell_idx].tail = (cell.tail >= 0 && (uint32_t)cell.tail < size) ? cell.tail : -1;
        cells[cell.cell_idx].seq_id = cell.seq_id;

        // Restore R/S tensor data — write each plane individually
        // (plane-interleaved layout; matches save_checkpoint's read order).
        size_t r_off = 0;
        size_t s_off = 0;
        for (int l = 0; l < n_layer; ++l) {
            if (r_l[l]) {
                for (uint32_t p = 0; p < recr_checkpoint_rows_per_cell; p++) {
                    size_t offset = ((size_t)cell.cell_idx + (size_t)p * (size_t)size) * r_row_bytes[l];
                    ggml_backend_tensor_set(r_l[l], cell.r_data.data() + r_off, offset, r_row_bytes[l]);
                    r_off += r_row_bytes[l];
                }
            }
            if (s_l[l]) {
                for (uint32_t p = 0; p < recr_checkpoint_rows_per_cell; p++) {
                    size_t offset = ((size_t)cell.cell_idx + (size_t)p * (size_t)size) * s_row_bytes[l];
                    ggml_backend_tensor_set(s_l[l], cell.s_data.data() + s_off, offset, s_row_bytes[l]);
                    s_off += s_row_bytes[l];
                }
            }
        }
    }

    // Recount used cells from restored metadata.
    uint32_t used_new = 0;
    for (auto & c : cells) {
        if (!c.is_empty()) {
            ++used_new;
        }
    }
    used = used_new;

    // Clear the snapshot so it is not applied twice.
    recr_checkpoint_cells.clear();
    recr_checkpoint_rows_per_cell = 0;
}

void llama_memory_recurrent::clear_checkpoint() {
    recr_checkpoint_cells.clear();
    recr_checkpoint_rows_per_cell = 0;
}

bool llama_memory_recurrent::expand(uint32_t new_mem_size) {
    // Always try to restore checkpoint if one exists, even if size doesn't change.
    // This handles the shrink(1)->expand(1) case where resize() cleared metadata.
    if (new_mem_size <= size) {
        if (!recr_checkpoint_cells.empty()) {
            restore_checkpoint();
        }
        return true;
    }

    bool ok = resize(new_mem_size);
    if (ok) {
        restore_checkpoint();
    }
    return ok;
}

bool llama_memory_recurrent::shrink(uint32_t new_mem_size) {
    if (new_mem_size == 0) {
        LLAMA_LOG_ERROR("%s: new_mem_size cannot be 0\n", __func__);
        return false;
    }

    if (new_mem_size >= size) {
        return true;
    }

    // Save recurrent state before shrinking so expand can restore it.
    save_checkpoint();

    return resize(new_mem_size);
}

bool llama_memory_recurrent::resize(uint32_t new_mem_size) {
    if (new_mem_size == size) {
        return true;
    }

    const int32_t n_layer = hparams.n_layer();
    const uint32_t old_size = size;
    const uint32_t n_copy = std::min(old_size, new_mem_size);

    struct buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u * n_layer * ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }
            ctx_map.emplace(buft, ctx);
            return ctx;
        }
        return it->second.get();
    };

    std::vector<ggml_tensor *> old_r_l = r_l;
    std::vector<ggml_tensor *> old_s_l = s_l;

    for (int i = 0; i < n_layer; i++) {
        if (!old_r_l[i] && !old_s_l[i]) {
            continue;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(old_r_l[i] ? old_r_l[i]->buffer : old_s_l[i]->buffer);
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create ggml context for resized rs cache\n", __func__);
            r_l = old_r_l;
            s_l = old_s_l;
            return false;
        }

        if (old_r_l[i]) {
            ggml_tensor * r = ggml_new_tensor_2d(ctx, old_r_l[i]->type, hparams.n_embd_r(), new_mem_size * (1 + n_rs_seq));
            ggml_format_name(r, "cache_r_l%d", i);
            r_l[i] = r;
        }
        if (old_s_l[i]) {
            ggml_tensor * s = ggml_new_tensor_2d(ctx, old_s_l[i]->type, hparams.n_embd_s(), new_mem_size * (1 + n_rs_seq));
            ggml_format_name(s, "cache_s_l%d", i);
            s_l[i] = s;
        }
    }

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> new_ctxs_bufs;
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate resized rs buffer\n", __func__);
            r_l = old_r_l;
            s_l = old_s_l;
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        new_ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    // Copy surviving cells from the old tensor to the new tensor,
    // respecting the plane-interleaved layout: primary cells first,
    // then snap_1 cells, then snap_2, etc.  Per-plane offsets differ
    // because old_size ≠ new_mem_size.
    if (n_copy > 0) {
        const uint32_t n_planes = 1 + n_rs_seq;
        std::vector<uint8_t> tmp;
        for (int i = 0; i < n_layer; i++) {
            if (old_r_l[i] && r_l[i]) {
                size_t row_bytes  = ggml_row_size(old_r_l[i]->type, hparams.n_embd_r());
                size_t copy_bytes = (size_t)n_copy * row_bytes;
                tmp.resize(copy_bytes);
                for (uint32_t p = 0; p < n_planes; p++) {
                    size_t old_offset = (size_t)p * old_size * row_bytes;
                    size_t new_offset = (size_t)p * new_mem_size * row_bytes;
                    ggml_backend_tensor_get(old_r_l[i], tmp.data(), old_offset, copy_bytes);
                    ggml_backend_tensor_set(r_l[i], tmp.data(), new_offset, copy_bytes);
                }
            }
            if (old_s_l[i] && s_l[i]) {
                size_t row_bytes  = ggml_row_size(old_s_l[i]->type, hparams.n_embd_s());
                size_t copy_bytes = (size_t)n_copy * row_bytes;
                tmp.resize(copy_bytes);
                for (uint32_t p = 0; p < n_planes; p++) {
                    size_t old_offset = (size_t)p * old_size * row_bytes;
                    size_t new_offset = (size_t)p * new_mem_size * row_bytes;
                    ggml_backend_tensor_get(old_s_l[i], tmp.data(), old_offset, copy_bytes);
                    ggml_backend_tensor_set(s_l[i], tmp.data(), new_offset, copy_bytes);
                }
            }
        }
    }

    ctxs_bufs = std::move(new_ctxs_bufs);
    cells.resize(new_mem_size);
    size = new_mem_size;

    uint32_t used_new = 0;
    for (auto & cell : cells) {
        cell.tail = -1;

        for (auto it = cell.seq_id.begin(); it != cell.seq_id.end();) {
            if (*it < 0 || (uint32_t) *it >= size) {
                LLAMA_LOG_WARN("%s: dropping seq_id %d after resize %u -> %u\n",
                        __func__, *it, old_size, new_mem_size);
                it = cell.seq_id.erase(it);
            } else {
                ++it;
            }
        }

        if (cell.seq_id.empty()) {
            cell.pos  = -1;
            cell.src  = -1;
            cell.src0 = -1;
            continue;
        }

        cell.src = -1;
        cell.src0 = -1;

        ++used_new;
    }

    used = used_new;

    // Restore self-referencing tails for cells that own themselves.
    // This is needed because we cleared all tails above, and when size doesn't
    // change (shrink(1)->expand(1)), restore_checkpoint may not be called.
    for (uint32_t i = 0; i < size; i++) {
        if (!cells[i].is_empty() && cells[i].tail == -1) {
            for (llama_seq_id sid : cells[i].seq_id) {
                if ((int32_t)sid >= 0 && (uint32_t)sid < size) {
                    cells[sid].tail = (int32_t)i;
                }
            }
        }
    }

    if (head >= size) {
        head = 0;
    }
    if (n >= size) {
        n = 0;
    }

    return true;
}

size_t llama_memory_recurrent::total_size() const {
    size_t size = 0;
    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_memory_recurrent::size_r_bytes() const {
    size_t size_r_bytes = 0;

    for (const auto & r : r_l) {
        if (r != nullptr) {
            size_r_bytes += ggml_nbytes(r);
        }
    }

    return size_r_bytes;
}

size_t llama_memory_recurrent::size_s_bytes() const {
    size_t size_s_bytes = 0;

    for (const auto & s : s_l) {
        if (s != nullptr) {
            size_s_bytes += ggml_nbytes(s);
        }
    }

    return size_s_bytes;
}

void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_data; // logical source row ranges
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            uint32_t rs_idx_cur = 0;

            if (n_rs_seq != 0) {
                if (seq_id != -1) {
                    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rs_idx.size());
                    rs_idx_cur = rs_idx[seq_id];
                } else {
                    bool has_rs_idx = false;
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        GGML_ASSERT(cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size());

                        const uint32_t seq_rs_idx = rs_idx[cell_seq_id];
                        if (!has_rs_idx) {
                            rs_idx_cur = seq_rs_idx;
                            has_rs_idx = true;
                        } else if (rs_idx_cur != seq_rs_idx) {
                            GGML_ABORT("cannot write shared recurrent state with different rollback indices");
                        }
                    }
                }
            }

            const uint32_t cell_id = rs_idx_cur * size + (cell.src >= 0 ? cell.src : (int32_t) i);
            if (cell_ranges_data.empty() || cell_ranges_data.back().second != cell_id) {
                cell_ranges_data.emplace_back(cell_id, cell_id + 1);
            } else {
                cell_ranges_data.back().second++;
            }

            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cell_ranges.size() > 1) {
        GGML_ABORT("cannot save/load multiple ranges of cells to/from device memory\n");
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : cell_ranges_data) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges_data);
}

void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // PARTIAL_ONLY / ON_DEVICE flags only affect the attention cache
    // at the hybrid layer; recurrent state is always fully restored.
    GGML_UNUSED(flags);

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    bool res = true;

    res = res && state_read_meta(io, cell_count, seq_id);
    res = res && state_read_data(io, cell_count);

    if (!res) {
        if (seq_id == -1) {
            clear(true);
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }

    if (n_rs_seq != 0) {
        if (seq_id == -1) {
            std::fill(rs_idx.begin(), rs_idx.end(), 0);
        } else {
            set_rs_idx(seq_id, 0);
        }

        // After restoring recurrent state (checkpoint restore / prompt cache
        // load), state_read_data only wrote the primary row (row 0).  Snapshot
        // rows 1..n_rs_seq still contain stale data from a previous sequence
        // or shrink/expand.  If a subsequent bounded rollback (rs_idx &gt; 0)
        // triggers s_copy to read from a snapshot row before any new data is
        // written there, the model would ingest cross-sequence recurrent state.
        //
        // Copy the primary row to every snapshot row so that every rollback
        // plane carries the just-restored state.  This is semantically safe:
        // a rollback within the next few decode steps can only rewind into
        // the restored state, which is exactly the last known-good state.
        if (seq_id != -1 && res) {
            int32_t cell_idx = cells[seq_id].tail;
            // Fallback: scan for a cell that owns this seq_id (tail may be -1
            // if state_read used the full-cache restore path with clear(true)).
            if (cell_idx < 0) {
                for (uint32_t i = 0; i < size; i++) {
                    if (cells[i].has_seq_id(seq_id)) {
                        cell_idx = (int32_t)i;
                        break;
                    }
                }
            }
            if (cell_idx >= 0 && (uint32_t)cell_idx < size) {
                cell_copy_primary_to_snapshots((uint32_t)cell_idx);
                LLAMA_LOG_DEBUG("%s: copied primary row to %u snapshot rows for cell %d after restore\n",
                               __func__, n_rs_seq, cell_idx);
            }
        }
    }
}

void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_memory_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the R tensors first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        // Write R tensor type
        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        // Write row size of R tensor
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * r_size_row;
            io.write_tensor(r_l[il], range.first * r_size_row, buf_size);
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write row size of S tensor
            const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            io.write(&s_size_row, sizeof(s_size_row));

            // Write each logical cell row range. With pending recurrent rollback,
            // the logical current state may live in a rollback snapshot plane.
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * s_size_row;
                io.write_tensor(s_l[il], range.first * s_size_row, buf_size);
            }
        }
    } else {
        // When S tensor is transposed, we also need the element size and get the element ranges from each row
        const uint32_t mem_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write element size
            const uint32_t s_size_el = ggml_type_size(s_l[il]->type);
            io.write(&s_size_el, sizeof(s_size_el));

            // Write GQA embedding size
            io.write(&n_embd_s, sizeof(n_embd_s));

            // For each row, we get the element values of each logical cell
            for (uint32_t j = 0; j < n_embd_s; ++j) {
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * mem_size) * s_size_el;
                    const size_t buf_size = range_size * s_size_el;
                    io.write_tensor(s_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_memory_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            ubatch.pos[i] = pos;
        }
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0] = &dest_seq_id;

        if (!find_slot(ubatch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == ubatch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= this->n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, this->n_seq_max);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = cells[seq_id].tail;
                if (tail != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        head = 0;
        used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_id = head + i;
        // make sure the recurrent states will keep their restored state
        cells[cell_id].src = cell_id;
    }

    return true;
}

bool llama_memory_recurrent::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer());
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (false != (bool) s_trans) {
        LLAMA_LOG_ERROR("%s: incompatible s transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (r_l[il] == nullptr) continue;

        // Read type of key
        int32_t r_type_i_ref;
        io.read(&r_type_i_ref, sizeof(r_type_i_ref));
        const int32_t r_type_i = (int32_t) r_l[il]->type;
        if (r_type_i != r_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, r_type_i, r_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t r_size_row_ref;
        io.read(&r_size_row_ref, sizeof(r_size_row_ref));
        const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        if (r_size_row != r_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, r_size_row, (size_t) r_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            io.read_tensor(r_l[il], head * r_size_row, cell_count * r_size_row);
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;

            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t s_size_row_ref;
            io.read(&s_size_row_ref, sizeof(s_size_row_ref));
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            if (s_size_row != s_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, s_size_row, (size_t) s_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                io.read_tensor(s_l[il], head * s_size_row, cell_count * s_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t s_size_el_ref;
            io.read(&s_size_el_ref, sizeof(s_size_el_ref));
            const size_t s_size_el = ggml_type_size(s_l[il]->type);
            if (s_size_el != s_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s element size (%zu != %zu, layer %d)\n", __func__, s_size_el, (size_t) s_size_el_ref, il);
                return false;
            }

            // Read state embedding size
            uint32_t n_embd_s_ref;
            io.read(&n_embd_s_ref, sizeof(n_embd_s_ref));
            if (n_embd_s != n_embd_s_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s embedding size (%u != %u, layer %d)\n", __func__, n_embd_s, n_embd_s_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_s; ++j) {
                    const size_t dst_offset = (head + j * size) * s_size_el;
                    io.read_tensor(s_l[il], dst_offset, cell_count * s_size_el);
                }
            }
        }
    }

    return true;
}

//
// llama_memory_recurrent_context
//

llama_memory_recurrent_context::llama_memory_recurrent_context(llama_memory_status status) : status(status) {}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), is_full(true) {
}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

llama_memory_recurrent_context::~llama_memory_recurrent_context() = default;

bool llama_memory_recurrent_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_recurrent_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is an update
    if (ubatches.empty()) {
        // recurrent cache never performs updates
        assert(status == LLAMA_MEMORY_STATUS_NO_UPDATE);

        return true;
    }

    mem->find_slot(ubatches[i_next]);

    return true;
}

llama_memory_status llama_memory_recurrent_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_recurrent_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_memory_recurrent_context::get_n_rs() const {
    return is_full ? mem->size : mem->n;
}

uint32_t llama_memory_recurrent_context::get_head() const {
    return is_full ? 0 : mem->head;
}

int32_t llama_memory_recurrent_context::get_rs_z() const {
    return is_full ? 0 : mem->rs_z;
}

uint32_t llama_memory_recurrent_context::get_size() const {
    return mem->size;
}

ggml_tensor * llama_memory_recurrent_context::get_r_l(int32_t il) const {
    return mem->r_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {
    return mem->s_l[il];
}

int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) {
        return src0;
    }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            // reset rollback idx
            mem->rs_idx[seq] = 0;
        }
    }
    return (int32_t)((size_t)idx * (size_t)mem->size) + src0;
}
