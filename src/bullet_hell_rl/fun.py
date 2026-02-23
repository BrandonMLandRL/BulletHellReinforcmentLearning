import math
def episodes_until_epsilon(x, start_eps=0.99, decay=0.999, warmup=200):
    return warmup + math.log(x / start_eps) / math.log(decay)

if __name__ == "__main__":
    print(episodes_until_epsilon(0.5))#882.7552393308725
    print(episodes_until_epsilon(0.01))#4792.82190709128