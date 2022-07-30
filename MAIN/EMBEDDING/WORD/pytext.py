import pytextrank

path_stage0 = "in.json"
path_stage1 = "o1.json"
path_stage2 = "o2.json"
path_stage3 = "o3.json"


with open(path_stage1, 'w') as f:
    for graf in pytextrank.parse_doc(pytextrank.json_iter(path_stage0)):
        f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))

graph, ranks = pytextrank.text_rank(path_stage1)
pytextrank.render_ranks(graph, ranks)


with open(path_stage2, 'w') as f:
    for rl in pytextrank.normalize_key_phrases(path_stage1, ranks):
        f.write("%s\n" % pytextrank.pretty_print(rl._asdict()))

kernel = pytextrank.rank_kernel(path_stage2)
with open(path_stage3, 'w') as f:
    for s in pytextrank.top_sentences(kernel, path_stage1):
        f.write(pytextrank.pretty_print(s._asdict()))
        f.write("\n")

phrases = ", ".join(set([p for p in pytextrank.limit_keyphrases(path_stage2, phrase_limit=3)]))
sent_iter = sorted(pytextrank.limit_sentences(path_stage3, word_limit=250), key=lambda x: x[1])
s = []

for sent_text, idx in sent_iter:
    s.append(pytextrank.make_sentence(sent_text))
graf_text = " ".join(s)
#print("**excerpts:** %s\n\n**keywords:** %s" % (graf_text, phrases,))
