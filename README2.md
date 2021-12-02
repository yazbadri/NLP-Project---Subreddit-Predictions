Robotics AI creates AI for robots that human's interact with. In order to build a better AI which can infer from context and other clues what text is referring to between two choices, they've tasked us with creating a Natural Lamguage Processing model that can do exactly that with text from two different subreddits.

In this project, we will be taking in the comments and titles from two subreddits: 49ers and Raiders, and creating a model that will try to predict from which subreddit each comment comes from. The goal is to see if we can create a model that can accurately predict the context of a comment without the knowledge of where the comment is coming from, by using natural language processing on the comments. We will be evaluating the model on the accuracy of its predictions and the $r^2$ score of it's features, which measures what percent of the variability in the predictions are explained by the features in the model. 

In this project we will be doing the following steps:

- Scraping data from two subreddits using PushShift API
- Exploring and cleaning the data
- Feature engineering
- Preprocessing the text data for use in our model
 - Tokenizing
 - Sentiment analysis
 - Lemmatizing
- Vectorizing text data
- Modeling including:
 - GridSearchCV
 - Multinomial Naïve Bayes Classifer
 - Decision Tree Classifier
 - ADA Boost Classifier
 - Linear SVC
 - Random Forest Classifier
- Final analysis and conclusion with recommendations


We'll use the top most common words to help and also hurt our models predictions(when we want to make it more challenging).

|model #|best_estimator|X_train_score|X_test_score|best_score|feature diff|
|--------|-------------|-------------|------------|----------|------------|
|model1	|TfidfVectorizer(), MultinomialNB()|0.673497|0.639344|0.595628|only TFID on posts|
|model2	|DecisionTreeClassifier()|0.91|0.71|0.73|only TFID on posts|
|model3	|AdaBoostClassifier(base_estimator=DecisionTreeClassifier())|1.0|1.0|1.0|corr + poly/corr|
|model4	|RandomForestClassifier()| 1.0|1.0|1.0 | corr + poly/corr|
|model5	|AdaBoostClassifier(base_estimator=LinearSVC())| 1.0|1.0|1.0|corr + poly/corr|
|model6	|(TfidfVectorizer(), MultinomialNB())| 0.55| 0.54|0.52|more stop words|
|model7	|DecisionTreeClassifier()| 0.85|0.65 |0.61| more stop words|
|model8	|AdaBoostClassifier(base_estimator=DecisionTreeClassifier())|0.871585|0.647541|0.649054|more stop words + corr|
|model9|AdaBoostClassifier(base_estimator=LinearSVC())|0.655738|0.684426|0.646286|more stop words + corr|
|model10|AdaBoostClassifier(base_estimator=DecisionTreeClassifier())|0.819672|0.606557|0.605302|only engineered features|
|model11|AdaBoostClassifier(base_estimator=DecisionTreeClassifier())|1.0|1.0|1.0|more stop words + corr + poly/corr|
|model12|AdaBoostClassifier(base_estimator=LinearSVC())|1.0|0.98|0.99|more stop words + corr + poly/corr|


- Our first model, the Multinomial Naïve Bayes had a fairly low accuracy as we only used the vectorized text column as features using a pipeline and grid. Keep in mind that the text was preprocessed before being vectorized.
- The second model, the DecisionTreeClassifier did quite a bit better once the params were adjusted and max depth was 5. The accuracy score was still not good enough, so we adjusted some features.
- Before running our third model, we used the most highly correlated words with the target 'subreddit' and then turned those columns into polynomials and used the most highly correlated of the new df with polynomials as features for our next 3 models.
- Our next 3 models all scored 100% or very close to!
- Considering this felt too easy, we decided to rerun our models but remove some of our top common words that are specifically team related. We added those words into our 'stop-words' parameter and reran the TFID vectorizer.
- For our sixth model, we ran the same exact model as our first but with our updated list of stop-words in our vectorizer. The score was a little lower as expected.
- We ran the exact same DecisionTreeClassifier model as our second model but with the new stop-words and again had lower accuracy scores as expected.
- For our next model, we used only the highly correlated words from our features as new features, and used an ADA Boost Classifier with our Decision Trees, and got a slight boost in our model's scores.
- We tried the same features and ADA boost Classifier but with a Linear SVC Model and got a very slight improvement in the accuracy scores.
- We wanted to see how much our engineered features (besides any text data/words) of word lengths and sentiment scores affected our model, so we ran a model with only those features and no words. The accuracy scores were unsurprisingly lower but surprisingly about 60% which tells us that they raise the accuracy by about 10 percentage points from our baseline of 50%.
- Finally we wanted to see if using the polynomial features would again boost our accuracy to the max, so we reran the correlation on polynomial features and used only the highest correlated features with the ADA boost Classifier with base estimator Decision Tree Classifier. We again got 100%, even with less features than our previous similar model and with our updated stop-words.

We created a model that accurately predicts from which subreddit a post comes from.
