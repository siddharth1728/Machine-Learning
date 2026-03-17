from scipy.stats import skew
data = [10,20,30,40,50,60,70,604,556,34]

print(skew(data, axis=0,bias=True))