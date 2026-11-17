// Native C++20 (-O3) kernels for Syntara: prefix-sharing token-index construction over a dense
// byte DFA, and masked argmax for constrained greedy decoding.
//
// The token index answers, for every DFA state s and vocabulary token t, "which state does the
// automaton reach after consuming all bytes of t from s?" (-1 if it dies). A naive build walks
// every byte of every token from every state: O(S * sum|t|). Walking the vocabulary in
// lexicographic order and resuming from the longest common prefix with the previous token
// (a depth-indexed state stack, i.e. an implicit trie DFS) makes each state's sweep cost
// proportional to the number of *distinct trie edges* instead of the total byte count.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr int32_t kAlphabet = 256;
}

extern "C" {

// table:     [num_states * 256] successor table, -1 = dead.
// bytes:     concatenated token bytes; token t occupies [offsets[t], offsets[t+1]).
// order:     token ids sorted lexicographically by their bytes.
// lcp:       lcp[k] = length of the common prefix of tokens order[k-1] and order[k] (lcp[0]=0).
// accepting: [num_states] 1 if accepting.
// out:       [num_states * vocab] next state per (state, token); EOS maps to the same state when
//            the state is accepting, otherwise -1. Zero-length non-EOS tokens are disallowed.
void syntara_build_token_index(
    const int32_t* __restrict__ table,
    int32_t num_states,
    const uint8_t* __restrict__ bytes,
    const int64_t* __restrict__ offsets,
    const int32_t* __restrict__ order,
    const int32_t* __restrict__ lcp,
    int32_t vocab,
    int32_t eos_id,
    const uint8_t* __restrict__ accepting,
    int32_t* __restrict__ out
) {
    int64_t max_len = 0;
    for (int32_t t = 0; t < vocab; ++t) {
        const int64_t len = offsets[t + 1] - offsets[t];
        if (len > max_len) {
            max_len = len;
        }
    }
    std::vector<int32_t> stack(static_cast<std::size_t>(max_len) + 1, -1);

    for (int32_t s = 0; s < num_states; ++s) {
        int32_t* row = out + static_cast<int64_t>(s) * vocab;
        stack[0] = s;
        int64_t valid = 0;  // stack[0..valid] hold live states for the previous token's prefix.
        for (int32_t k = 0; k < vocab; ++k) {
            const int32_t t = order[k];
            const int64_t begin = offsets[t];
            const int64_t len = offsets[t + 1] - begin;
            if (t == eos_id) {
                row[t] = accepting[s] ? s : -1;
                continue;
            }
            if (len == 0) {
                row[t] = -1;
                continue;
            }
            const int64_t shared = lcp[k];
            if (shared > valid) {
                // The previous token died inside the shared prefix, so this one dies too.
                row[t] = -1;
                continue;
            }
            int64_t depth = shared;
            bool dead = false;
            while (depth < len) {
                const int32_t nxt = table[static_cast<int64_t>(stack[depth]) * kAlphabet + bytes[begin + depth]];
                if (nxt < 0) {
                    dead = true;
                    break;
                }
                stack[depth + 1] = nxt;
                ++depth;
            }
            valid = depth;
            row[t] = dead ? -1 : stack[len];
        }
    }
}

// Greedy constrained decoding step: argmax over logits restricted to tokens whose next state is
// >= 0. Returns -1 when no token is allowed. Ties resolve to the smallest token id.
int32_t syntara_masked_argmax(
    const double* __restrict__ logits,
    const int32_t* __restrict__ next_row,
    int32_t vocab
) {
    int32_t best = -1;
    double best_val = -std::numeric_limits<double>::infinity();
    for (int32_t t = 0; t < vocab; ++t) {
        if (next_row[t] >= 0 && (best < 0 || logits[t] > best_val)) {
            best = t;
            best_val = logits[t];
        }
    }
    return best;
}

// Count allowed tokens per state (mask density diagnostics).
void syntara_count_allowed(
    const int32_t* __restrict__ index,
    int32_t num_states,
    int32_t vocab,
    int32_t* __restrict__ counts
) {
    for (int32_t s = 0; s < num_states; ++s) {
        const int32_t* row = index + static_cast<int64_t>(s) * vocab;
        int32_t c = 0;
        for (int32_t t = 0; t < vocab; ++t) {
            c += row[t] >= 0 ? 1 : 0;
        }
        counts[s] = c;
    }
}

}  // extern "C"
