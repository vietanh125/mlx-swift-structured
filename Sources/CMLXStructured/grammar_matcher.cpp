#include "mlx_structured/error_handler.h"
#include "mlx_structured/grammar_matcher.h"
#include <dlpack/dlpack.h>
#include <limits>
#include <xgrammar/matcher.h>

using namespace xgrammar;

extern "C" void* grammar_matcher_new(void* compiled_grammar) {
    try {
        auto* compiled_grammar_ptr = static_cast<CompiledGrammar*>(compiled_grammar);
        auto* grammar_matcher_ptr = new GrammarMatcher(*compiled_grammar_ptr);
        return grammar_matcher_ptr;
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

extern "C" bool grammar_matcher_fill_next_token_bitmask(
    void* grammar_matcher,
    void* next_token_bitmask
) {
    try {
        auto* grammar_matcher_ptr = static_cast<GrammarMatcher*>(grammar_matcher);
        auto* next_token_bitmask_ptr = static_cast<DLTensor*>(next_token_bitmask);
        return grammar_matcher_ptr->FillNextTokenBitmask(next_token_bitmask_ptr);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return false;
    }
}

extern "C" bool grammar_matcher_fill_next_token_dense_mask(
    void* grammar_matcher,
    void* next_token_bitmask,
    float* next_token_dense_mask,
    size_t next_token_dense_mask_len
) {
    try {
        auto* grammar_matcher_ptr = static_cast<GrammarMatcher*>(grammar_matcher);
        auto* next_token_bitmask_ptr = static_cast<DLTensor*>(next_token_bitmask);
        if (!grammar_matcher_ptr->FillNextTokenBitmask(next_token_bitmask_ptr)) {
            return false;
        }

        auto* data_ptr = reinterpret_cast<const uint32_t*>(next_token_bitmask_ptr->data);
        constexpr size_t bits_per_block = sizeof(uint32_t) * 8;
        const float negative_infinity = -std::numeric_limits<float>::infinity();
        const size_t block_count = (next_token_dense_mask_len + bits_per_block - 1) / bits_per_block;
        for (size_t block_index = 0; block_index < block_count; ++block_index) {
            const uint32_t block = data_ptr[block_index];
            const size_t base = block_index * bits_per_block;
            const size_t limit = std::min(bits_per_block, next_token_dense_mask_len - base);
            for (size_t bit = 0; bit < limit; ++bit) {
                next_token_dense_mask[base + bit] = (block & (uint32_t(1) << bit))
                    ? 0.0f
                    : negative_infinity;
            }
        }
        return true;
    } catch (const std::exception& e) {
        catch_error(e.what());
        return false;
    }
}

extern "C" bool grammar_matcher_accept_token(
    void* grammar_matcher,
    int32_t token_id
) {
    try {
        auto* grammar_matcher_ptr = static_cast<GrammarMatcher*>(grammar_matcher);
        return grammar_matcher_ptr->AcceptToken(token_id);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return false;
    }
}

extern "C" int8_t grammar_matcher_accept_token_status(
    void* grammar_matcher,
    int32_t token_id
) {
    try {
        auto* grammar_matcher_ptr = static_cast<GrammarMatcher*>(grammar_matcher);
        if (grammar_matcher_ptr->IsTerminated()) {
            return -1;
        }
        return grammar_matcher_ptr->AcceptToken(token_id) ? 1 : 0;
    } catch (const std::exception& e) {
        catch_error(e.what());
        return 0;
    }
}

extern "C" bool grammar_matcher_is_terminated(void* grammar_matcher) {
    try {
        auto* grammar_matcher_ptr = static_cast<GrammarMatcher*>(grammar_matcher);
        return grammar_matcher_ptr->IsTerminated();
    } catch (const std::exception& e) {
        catch_error(e.what());
        return false;
    }
}



extern "C" void grammar_matcher_reset(void* grammar_matcher) {
    try {
        auto* grammar_matcher_ptr = static_cast<GrammarMatcher*>(grammar_matcher);
        grammar_matcher_ptr->Reset();
    } catch (const std::exception& e) {
        catch_error(e.what());
        return;
    }
}

extern "C" void grammar_matcher_free(void* grammar_matcher) {
    if (grammar_matcher) {
        delete static_cast<GrammarMatcher*>(grammar_matcher);
    }
}
