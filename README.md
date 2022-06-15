# Sentiment-Analysis-of-Titled-Hotel-Reviews-with-Shallow-Gated-Recurrent-Units

### PROJECT CONCEPT
This is a research project with the aim of providing a relatively simple, computationally inexpensive approach to sentiment analysis of hotel reviews. To achieve this goal, a well-optimised single hidden-layer GRU was proposed and its performance was compared to that of a simple vector machine (SVM) model.<br><br>

### DATASET
Hotel review data used in this research was collected from Opin Rank datasets (Ganesan and Zhai, 2012) comprising titled comments from guests about hotels in many cities worldwide. Only [customer reviews of London hotels](https://user-images.githubusercontent.com/76821049/173886937-80017dff-d71f-4d3d-93e0-1614aedc7ced.png) were used in this study. Due to the absence of sentiment labels, a human annotator was tasked with assigning sentiment labels (-1 for negative reviews and 1 for positive reviews) to the reviews/documents by the emotional meaning of the keywords present in each document. Keywords were determined by selecting only matching words between each document’s topic and its body of text. [Here](https://user-images.githubusercontent.com/76821049/173887463-56e1c396-7f28-4a85-82a7-8ab835d34619.png), you can find the labelled dataset.<br>

