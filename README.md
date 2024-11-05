# cmu_ml2
this is for machine learning 2. 

Team 1: Nikhil Dittakavi, Srinivasa Cheemakurthi, Samira Shah, Jessica Waissmann, Elaaf Khan

Project Update 1
Scenario: We are the marketing team for Nike. As part of our march madness campaigns we sponsor
teams and provide them with shoes & gear to wear during their games. For the 2025 March
Madness Season, we are competing with Adidas & other companies for sponsorships of the top
teams. In order to maximize return on investment, we want to be able to predict the top teams that
will win in 2025 and allocate the most time & budget to obtain said sponsorships.
Business Problem Summary:
Using predictive analytics on the March Madness 2024 dataset, we aim to forecast NCAA team
performance and offer strategic investment insights for sponsorship of the best teams
Problem Using Solved ML Details: Teams in Kaggle have solved 2 use cases like Bracketology and
Upsets wins.
Data Set Summary: We have picked the data from Kaggle.
Business Problem:
We are using the March Madness 2024 dataset to predict team performance in the NCAA
tournament and offer investment strategies based on these predictions. March Madness draws over
100 million viewers and 70 million bracket participants annually. By applying predictive analytics, we
aim to provide insights on top-performing teams, potential upsets, and key trends, helping Nike
decide on top sponsorship investments.
Why It’s Important:
This problem is interesting because it offers maximization & optimization of return on investment
for Nike:
● Sponsorship & Media: Predicting top-performing teams allows Nike to optimize
sponsorships around high-viewership games.
● Merchandising: opportunities to create and sell new merchandising around top players &
teams ( e.g. creating a nike shirt for the UConn who won 2024 March madness)
Financial Impact:
Solving this problem helps businesses make money by:
● Optimizing sponsorship and media deals
● Boosting merchandise
This type of business problem—predicting outcomes in March Madness using machine learning—has
been addressed using various approaches. One example is a Kaggle project, where the problem of
predicting NCAA tournament outcomes was solved using machine learning models to predict team
performance. In the project titled "Bracketology: Predicting March Madness" by Nishaan Amin, a
variety of models such as Logistic Regression, Random Forest, and Gradient Boosting were applied to
predict game outcomes based on historical team statistics and tournament results. These models
helped create optimized brackets and investment strategies for betting and fan engagement, our
work builds on such predictive analytics, offering businesses data-driven insights to optimize betting
odds, sponsorship deals, and fan merchandise sales, further monetizing the NCAA tournament's
massive audience.
In the project, XGBoost (XGBRegressor) was used to predict March Madness outcomes.
Key Characteristics:
● Gradient Boosting: XGBoost builds decision trees sequentially, with each new tree
correcting errors from the previous one, improving accuracy.
● Regularization: L1 and L2 regularization reduces overfitting, crucial for complex sports data.
● Handling Missing Data: XGBoost deals with missing values automatically during training.
● Feature Importance: It ranks features (e.g., team stats) based on their contribution to
predictions.
Advantages:
● High Accuracy: Ideal for complex, non-linear data like tournament outcomes.
● Efficiency: Fast and optimized for large datasets.
● Prevents Overfitting: Thanks to built-in regularization.
Disadvantages:
● Complex Tuning: Requires careful hyperparameter optimization.
● Less Interpretable: Harder to explain individual predictions compared to simpler models.
Why Used:
XGBoost was chosen for its ability to model complex relationships in sports data, offering high
accuracy and efficient performance for March Madness predictions.
We could use the dataset from the Kaggle March Madness Dataset -
https://www.kaggle.com/datasets/nishaanamin/march-madness-data mentioned in the Kaggle
competition to build a proof-of-concept (POC) ML system for predicting tournament outcomes. This
dataset includes historical NCAA tournament results, team statistics, and other relevant features
that can be used to train our machine learning model.
Additionally, we could also use the NCAA Basketball Historical Data from the Sports Reference
website - https://www.sports-reference.com/cbb/, which provides comprehensive statistics on
college basketball teams and players over the years. This dataset can be downloaded at Sports
Reference.
Both datasets will be valuable for developing predictive models based on historical performance and
team metrics.
March Madness dataset consists of 25 files:
1. 538 Ratings.csv: Team ratings based on FiveThirtyEight's statistical models.
2. Barttorvik Away-Neutral.csv: Performance metrics for teams in away and neutral games.
3. Barttorvik Away.csv: Performance metrics for teams in away games.
4. Barttorvik Home.csv: Performance metrics for teams in home games.
5. Barttorvik Neutral.csv: Performance metrics for teams in neutral site games.
6. Coach Results.csv: Win-loss statistics and performance records for coaches.
7. Conference Results.csv: Aggregated results from various conferences.
8. Conference Stats.csv: Overall statistics for each conference (scoring, rebounds, etc.).
9. Conference Stats Home.csv: Home performance metrics for conference teams.
10. Conference Stats Away.csv: Away performance metrics for conference teams.
11. Conference Stats Neutral.csv: Neutral site performance metrics for conference teams.
12. Conference Stats Away Neutral.csv: Away performance metrics in neutral venues.
13. Heat Check Tournament Index.csv: Composite index reflecting team performance before
the tournament.
14. KenPom Barttorvik.csv: Advanced statistics and ratings from Ken Pomeroy.
15. Preseason Votes.csv: Preseason rankings and expectations for teams.
16. Public Picks.csv: Data on public bracket predictions and selections.
17. Resumes.csv: Team resumes for tournament selection with key wins and losses.
18. Seed Results.csv: Historical performance data based on team seeding.
19. Shooting Splits.csv: Shooting statistics for teams (field goal percentages).
20. Team Results.csv: Comprehensive results for each team across seasons.
21. Tournament Locations.csv: Locations of tournament games.
22. Tournament Matchups.csv: Historical matchup data between teams.
23. Tournament Simulation.csv: Simulated outcomes based on historical data.
24. Upset Count.csv: Number of upsets in tournaments over the years.
25. Upset Seed Info.csv: Seeds of teams that caused upsets.
These datasets provide a comprehensive foundation for analysing NCAA tournament outcomes and
team performance.
