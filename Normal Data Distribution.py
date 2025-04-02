import matplotlib.pyplot as plt
import numpy as np

exam_scores = np.random.normal(70, 10, 100)

plt.hist(exam_scores, bins=20, edgecolor='black')
plt.title(" Distribution Exam Scores")
plt.xlabel("Scores") 
plt.ylabel("Frequency")
plt.show()
