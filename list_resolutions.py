for i in range(100):
    w = 16*i
    h = 9*i
    if w % 32 == 0 and h % 32 == 0:
        print(str(w) + ':' + str(h))
