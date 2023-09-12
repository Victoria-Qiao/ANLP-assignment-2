# ANLP-assignment-2

Accelerated Natural Language Processing (2020) assignment 2: Exploring distributional similarity in Twitter

In this report, how different context vectors and similarity computation methods influence the similarities between words was investigated. We picked six groups of words. For each group, there is a reference word, and there were three words for each of three categories: similar, moderately similar and not similar.

Four methods were evaluated. Two of the methods are PPMI and t-test, which are used to compute context vectors of words; the other two methods are cosine similarity and euclidean distance, which are similarity measures to compare two context vectors of two words.

By combining the two methods of calculating the word vector and the method of calculating the similarity, we obtain four methods of calculating the similarity between two words and apply these methods to a given word list.

asgn2.py is the code to do the experiment. In this code, we implemented functions of **t_test**, **euclid_sim**, **cos_sim**, **create_ppmi_vectors** and **create_t_test_vectors**.
