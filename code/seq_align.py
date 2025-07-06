def align_sequences(tok1, tok2):
    # 初始化结果列表
    align1 = []
    align2 = []
    
    i = 0  # tok1的索引
    j = 0  # tok2的索引
    
    while i < len(tok1) and j < len(tok2):
        # 如果token完全相同
        if tok1[i] == tok2[j]:
            align1.append(i)
            align2.append(j)
            i += 1
            j += 1
        else:
            # 尝试合并检查
            # 1. 检查tok1的两个或三个token是否等于tok2的一个token  "Ġ", "▁" 
            if i + 2 < len(tok1) and ''.join([tok1[i], tok1[i+1], tok1[i+2]]).replace('▁', '').replace('Ġ', '') == tok2[j].replace('▁', '').replace('Ġ', ''):
                # align1.append((i, i+1, i+2))
                align1.append(i+2)
                align2.append(j)
                i += 3
                j += 1
            elif i + 1 < len(tok1) and ''.join([tok1[i], tok1[i+1]]).replace('▁', '').replace('Ġ', '') == tok2[j].replace('▁', '').replace('Ġ', ''):
                # align1.append((i, i+1))
                align1.append(i+1)
                align2.append(j)
                i += 2
                j += 1
            # 2. 检查tok2的两个或三个token是否等于tok1的一个token
            elif j + 2 < len(tok2) and tok1[i].replace('▁', '').replace('Ġ', '') == ''.join([tok2[j], tok2[j+1], tok2[j+2]]).replace('▁', '').replace('Ġ', ''):
                align1.append(i)
                # align2.append((j, j+1, j+2))
                align2.append(j+2)
                i += 1
                j += 3
            elif j + 1 < len(tok2) and tok1[i].replace('▁', '').replace('Ġ', '') == ''.join([tok2[j], tok2[j+1]]).replace('▁', '').replace('Ġ', ''):
                align1.append(i)
                # align2.append((j, j+1))
                align2.append(j+1)
                i += 1
                j += 2
            else:
                # 如果都不匹配，可能需要更复杂的处理
                # print(f"无法对齐的位置: tok1[{i}]={tok1[i]}, tok2[{j}]={tok2[j]}")
                i += 1
                j += 1
    
    assert len(align1) == len(align2)
    new_align1 = []
    new_align2 = []
    for _a1, _a2 in zip(align1, align2):
        if tok1[_a1] == tok2[_a2] or tok1[_a1].replace('▁', '').replace('Ġ', '') == tok2[_a2].replace('▁', '').replace('Ġ', ''):
            new_align1.append(_a1)
            new_align2.append(_a2)

    return new_align1, new_align2


def new_align_sequences(tok1, tok2):
    p1 = -1
    p2 = -1
    rtok1 = [i.replace('▁', '').replace('Ġ', '') for i in tok1]
    rtok2 = [i.replace('▁', '').replace('Ġ', '') for i in tok2]

    align1 = []
    align2 = []

    history_seq1 = ""
    history_seq2 = ""

    while p1 < len(rtok1)-1 and p2 < len(rtok2)-1:
        if history_seq1 == history_seq2 and rtok1[p1+1] == rtok2[p2+1]:
            history_seq1 += rtok1[p1+1]
            history_seq2 += rtok2[p2+1]
            align1.append(p1+1)
            align2.append(p2+1)
            p1 += 1
            p2 += 1
        elif len(history_seq1) > len(history_seq2):
            history_seq2 += rtok2[p2+1]
            p2 += 1
        elif len(history_seq1) < len(history_seq2):
            history_seq1 += rtok1[p1+1]
            p1 += 1
        else:
            history_seq1 += rtok1[p1+1]
            history_seq2 += rtok2[p2+1]
            p1 += 1
            p2 += 1

    for i, j in zip(align1, align2):
        assert tok1[i] == tok2[j]
    return align1, align2

seq1 = ["my", "boy", "fr", "ie", "nd", "is", "a", "good", "peop", "le", "!"]
seq2 = ["my", "boyfri", "end",  "is", "a", "good", "people", "!"]

# a1, a2 = align_sequences(seq1, seq2)
a1, a2 = new_align_sequences(seq1, seq2)


print(a1, a2)

for i, j in zip(a1, a2):
    print(seq1[i], seq2[j])
