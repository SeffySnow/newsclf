from newsclf import NewsClassifier, predict

clf = NewsClassifier(select="best", metric="test_macro_f1", max_len=256)
label, conf = clf.predict("Streaming platform renews hit fantasy series for a third season")
print(label, conf)

# one-shot helper
predict("UK lawmakers approve new budget amid inflation concerns.", select="best", max_len=256)
