"""
Pure-Python phrase-based statistical MT (Prakrit -> Hindi).

Moses is not installed on this machine and is awkward to set up on Windows, so
this reimplements the core pipeline:

  1. IBM Model 1 word alignment (EM), both directions
  2. Symmetrization: intersection + union points that touch an unaligned word
  3. Consistent phrase-pair extraction (Koehn et al. 2003), length <= 3
  4. Phrase scores: relative frequency p(e|f), p(f|e) and lexical weight lex(e|f)
  5. Interpolated bigram LM over the training targets
  6. Monotone beam-search decoder (no reordering), copy-through or drop for OOV
  7. Weight tuning: coordinate grid search on a held-out dev split (see run_smt_cv.py)

Not implemented: reordering model, MERT, the full Moses feature set.
"""
import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[ऀ-ॿ]+|[A-Za-z]+|\d+|[^\s]")
PUNCT_RE = re.compile(r" ([।,?!.;:])")
MAX_PHRASE = 3
NBEST = 5
BEAM = 60
OOV_COST = -3.0
NULL = "<NULL>"

# Default feature weights. run_smt_cv.py tunes these per fold on a dev split.
DEFAULT_WEIGHTS = {
    "tm": 1.0,       # phrase-table log-probs p(e|f) * p(f|e)
    "lex": 0.5,      # lexical weight lex(e|f)
    "lm": 0.5,       # bigram LM
    "word": 0.3,     # reward per output word
    "phrase": -0.2,  # per-phrase penalty
}


def tokenize(s):
    return TOKEN_RE.findall(s)


def detokenize(tokens):
    return PUNCT_RE.sub(r"\1", " ".join(tokens))


def ibm1(pairs, iters=10):
    """t[(e, f)] = p(e | f) for (src_tokens, tgt_tokens) pairs. NULL is a source word."""
    e_vocab = {e for _, es in pairs for e in es}
    uniform = 1.0 / len(e_vocab)
    t = defaultdict(lambda: uniform)
    for _ in range(iters):
        count = defaultdict(float)
        total = defaultdict(float)
        for fs, es in pairs:
            fs = [NULL] + fs
            for e in es:
                denom = sum(t[(e, f)] for f in fs)
                for f in fs:
                    c = t[(e, f)] / denom
                    count[(e, f)] += c
                    total[f] += c
        t = defaultdict(lambda: uniform, {k: v / total[k[1]] for k, v in count.items()})
    return t


def align_points(fs, es, t):
    """Best-link alignment: for each target word, argmax over source words."""
    links = set()
    for j, e in enumerate(es):
        best, best_p = None, 0.0
        for i, f in enumerate(fs):
            p = t.get((e, f), 0.0)
            if p > best_p:
                best, best_p = i, p
        if best is not None:
            links.add((best, j))
    return links


def symmetrize(pairs, t_fe, t_ef):
    """t_fe: p(e|f) (src->tgt model), t_ef: p(f|e) (tgt->src model)."""
    out = []
    for fs, es in pairs:
        fwd = align_points(fs, es, t_fe)
        rev = {(i, j) for (j, i) in align_points(es, fs, t_ef)}
        inter = fwd & rev
        union = fwd | rev
        aligned_f = {i for i, _ in inter}
        aligned_e = {j for _, j in inter}
        grown = set(inter)
        for i, j in union - inter:
            if i not in aligned_f or j not in aligned_e:
                grown.add((i, j))
                aligned_f.add(i)
                aligned_e.add(j)
        out.append(grown)
    return out


def extract_phrases(fs, es, links):
    """Yield (src_span, tgt_span, src_tokens, tgt_tokens) for consistent phrase pairs."""
    n = len(fs)
    for i1 in range(n):
        for i2 in range(i1, min(n, i1 + MAX_PHRASE)):
            tgt = [j for i, j in links if i1 <= i <= i2]
            if not tgt:
                continue
            j1, j2 = min(tgt), max(tgt)
            if j2 - j1 + 1 > MAX_PHRASE:
                continue
            # consistency: no link from outside [i1,i2] into [j1,j2]
            if any(j1 <= j <= j2 and not (i1 <= i <= i2) for i, j in links):
                continue
            yield (i1, i2), (j1, j2)


def lex_weight(fs, es, links, src_span, tgt_span, t):
    """Koehn's lexical weight lex(e|f): product over target words of the mean
    word translation probability over the aligned source words (NULL if none)."""
    (i1, i2), (j1, j2) = src_span, tgt_span
    score = 1.0
    for j in range(j1, j2 + 1):
        aligned = [i for i, jj in links if jj == j and i1 <= i <= i2]
        if aligned:
            score *= sum(t[(es[j], fs[i])] for i in aligned) / len(aligned)
        else:
            score *= t[(es[j], NULL)]
    return score


class PhraseModel:
    def __init__(self, pairs):
        t_fe = ibm1(pairs)                              # p(e|f)
        t_ef = ibm1([(es, fs) for fs, es in pairs])     # p(f|e)
        self.t_fe = t_fe
        links_all = symmetrize(pairs, t_fe, t_ef)

        pair_c = Counter()
        src_c = Counter()
        tgt_c = Counter()
        lex_sum = defaultdict(float)
        for (fs, es), links in zip(pairs, links_all):
            for s_span, t_span in extract_phrases(fs, es, links):
                f = tuple(fs[s_span[0]:s_span[1] + 1])
                e = tuple(es[t_span[0]:t_span[1] + 1])
                pair_c[(f, e)] += 1
                src_c[f] += 1
                tgt_c[e] += 1
                lex_sum[(f, e)] += lex_weight(fs, es, links, s_span, t_span, t_fe)

        self.table = defaultdict(list)  # src phrase -> [(tgt phrase, feature dict)]
        for (f, e), c in pair_c.items():
            feats = {
                "tm": math.log(c / src_c[f]) + math.log(c / tgt_c[e]),
                "lex": math.log(max(lex_sum[(f, e)] / c, 1e-12)),
            }
            self.table[f].append((e, feats))
        for f in self.table:
            self.table[f].sort(key=lambda x: -x[1]["tm"])
            self.table[f] = self.table[f][:NBEST]

        self.lm = BigramLM([es for _, es in pairs])

    def translate(self, src_tokens, w=DEFAULT_WEIGHTS, copy_oov=True):
        n = len(src_tokens)
        beams = [[] for _ in range(n + 1)]
        beams[0] = [(0.0, (), 0)]  # (score, tgt tokens so far, coverage)
        for i in range(n):
            beams[i] = sorted(beams[i], key=lambda h: -h[0])[:BEAM]
            for score, out, _ in beams[i]:
                prev = out[-1] if out else "<s>"
                for L in range(1, MAX_PHRASE + 1):
                    if i + L > n:
                        break
                    f = tuple(src_tokens[i:i + L])
                    for e, feats in self.table.get(f, []):
                        lm = self.lm.score(prev, e[0])
                        lm_total = lm + sum(self.lm.score(a, b) for a, b in zip(e, e[1:]))
                        s = (score + w["tm"] * feats["tm"] + w["lex"] * feats["lex"]
                             + w["lm"] * lm_total + w["word"] * len(e) + w["phrase"])
                        beams[i + L].append((s, out + e, i + L))
                f = (src_tokens[i],)
                if f not in self.table:
                    if copy_oov:
                        lm = self.lm.score(prev, src_tokens[i])
                        s = score + OOV_COST + w["lm"] * lm + w["word"]
                        beams[i + 1].append((s, out + (src_tokens[i],), i + 1))
                    else:
                        beams[i + 1].append((score + OOV_COST, out, i + 1))
        final = beams[n]
        if not final:
            return ""
        best = max(final, key=lambda h: h[0] + w["lm"] * self.lm.score(h[1][-1] if h[1] else "<s>", "</s>"))
        return detokenize(list(best[1]))


class BigramLM:
    """Interpolated bigram with unigram backoff. Trained on target token sequences."""

    def __init__(self, sents, lam=0.7):
        self.lam = lam
        self.uni = Counter()
        self.bi = Counter()
        self.ctx = Counter()
        for toks in sents:
            seq = ["<s>"] + list(toks) + ["</s>"]
            for a, b in zip(seq, seq[1:]):
                self.bi[(a, b)] += 1
                self.ctx[a] += 1
            self.uni.update(seq)
        self.total = sum(self.uni.values())
        self.V = len(self.uni) + 1

    def score(self, prev, word):
        p_uni = (self.uni.get(word, 0) + 0.5) / (self.total + 0.5 * self.V)
        p_bi = self.bi.get((prev, word), 0) / self.ctx[prev] if self.ctx.get(prev, 0) else 0.0
        return math.log(self.lam * p_bi + (1 - self.lam) * p_uni)
