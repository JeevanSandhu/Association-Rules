data = open('murderers.csv').read().split('\n')
data = [asdf.split(',') for asdf in data]
data.pop()

attr = [[asdf[9], asdf[10], asdf[12], asdf[13]] for asdf in data]
del attr[0]

a = [asdf[0] for asdf in attr]
b = [asdf[1] for asdf in attr]
c = [asdf[2] for asdf in attr]
d = [asdf[3] for asdf in attr]
set(a)
set(b)
set(c)
set(d)

for i, asdf in enumerate(attr):
    if asdf[2]=='Victim under the age of 18':
        attr[i][2]='VU18'
    elif asdf[2]=='Victim over the age of 18':
        attr[i][2]='VO18'

for i,asdf in enumerate(attr):
    if asdf[3]=='Child Murderer':
        attr[i][3]='CMurderer'
    elif asdf[3]=='Murderer':
        attr[i][3]='Murderer'
    elif asdf[3]=='Sexual Predator':
        attr[i][3]='SPredator'
    elif asdf[3]=='Sexually Dangerous':
        attr[i][3]='SDangerous'
    elif asdf[3]=='Sexually Violent':
        attr[i][3]='SViolent'
        
for i,asdf in enumerate(attr):
    if asdf[1]=="F":
        attr[i][1]="Female"   
    else:
        attr[i][1]="Male"     

s=''
for asdf in attr:
    s = s + asdf[0] + ", " + asdf[1] + ", " + asdf[2] + ", " + asdf[3] + "\n"

import re
import Orange

word = re.compile("\w+")

all_items = set(word.findall(s))
domain = Orange.data.Domain([])
domain.add_metas({Orange.orange.newmetaid(): Orange.feature.Continuous(n) for n in all_items}, True)

data = Orange.data.Table(domain)
for e in s.splitlines():
    ex = Orange.data.Instance(domain)
    for m in re.findall("\w+", e):
        ex[m] = 1
    data.append(ex)

rules = Orange.associate.AssociationRulesSparseInducer(data, support=0.6)


Orange.associate.sort(rules, ms=['support'])
# Orange.associate.sort(rules, ms=['confidence'])

# only 5 rules
# print "%4s %4s  %s" % ("Supp", "Conf", "Rule")
# for r in rules[:5]:
#     print "%4.1f %4.1f  %s" % (r.support, r.confidence, r)

# all rules
print "%4s %4s  %s" % ("Supp", "Conf", "Rule")
for r in rules:
    print "%4.1f %4.1f  %s" % (r.support, r.confidence, r)

print "\n\n"
ind = Orange.associate.AssociationRulesSparseInducer(support=0.5, storeExamples = True)
itemsets = ind.get_itemsets(data)
print "support    freq item set"
for itemset, tids in itemsets:
    print "(%4.2f) %s" % (len(tids)/float(len(data)),
                          " ".join(data.domain[item].name for item in itemset))

list_attr = ['CMurderer', 'Murderer', 'SPredator', 'SDangerous', 'SViolent']

pred = []
for d in data:
    flag=0
    for rule in rules:
        if rule.applies_left(d):
            right_side = list(rule.right.get_metas(str))
            asdf = [bla for bla in right_side if bla in list_attr]
            # print asdf
            if(len(asdf)>0):
                pred.append(asdf[0])
                # print str(d) + " --> " + str(asdf)
                flag=1
                break
    if flag==0:
        pred.append("NONE")

actual_labels = [asdf[3] for asdf in attr]
predicted_labels = pred


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, hamming_loss, jaccard_similarity_score, matthews_corrcoef, zero_one_loss

accuracy_score = accuracy_score(actual_labels, predicted_labels)
print("\n\nAccuracy {} %".format(round(accuracy_score*100,3)))

confusion_matrix = confusion_matrix(actual_labels, predicted_labels)
print("\n\nConfusion Matrix: \n\n {}".format(confusion_matrix))

classification_report = classification_report(actual_labels, predicted_labels)
print("\n\nClassification Scores: \n\n {}".format(classification_report))

hamming_loss = hamming_loss(actual_labels, predicted_labels)
print("\n\nHamming Loss {}".format(hamming_loss))

jaccard_similarity_score = jaccard_similarity_score(actual_labels, predicted_labels)
print("\n\nJaccard Similarity Score {}".format(jaccard_similarity_score))

zero_one_loss = zero_one_loss(actual_labels, predicted_labels)
print("\n\nZero-One Loss {}".format(zero_one_loss))