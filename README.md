# Digit Recognizer

A "from-the-ground-up" implementation of the Kaggle [Digit Recognizer] problem.

This is a Python implementation of the initial hand-coded C# solution provided by Mathias Brandewinder in his book, _[Machine Learning Projects for .NET Developers]_.

From his description:

>What we have is a dataset of 50,000 images. Each image is a single digit, written
down by a human, and scanned in 28 x 28 pixels resolution, encoded in grayscale, with each pixel taking one of 256 possible shades of gray, from full white to full black. For each scan, we also know the correct answer, that is, what number the human wrote down. This dataset is known as the training set. Our goal now is to write a program that will learn from the training set and use that information to make predictions for images it has never seen before: is it a zero, a one, and so on. 
>
>Technically, this is known as a classification problem: Our goal is to separate images between known "categories," a.k.a. the classes (hence the word "classification"). In this case, we have ten classes, one for each single digit from 0 to 9. 

The solution provided here does not use any libraries, hence the designation "from-the-ground-up." 

[Machine Learning Projects for .NET Developers]: <https://www.apress.com/gb/book/9781430267676>
[Digit Recognizer]: <https://www.kaggle.com/c/digit-recognizer/overview>