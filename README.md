Spam messages are unsolicited or unwanted emails/messages sent in bulk to users. Detecting spam emails automatically helps prevent unnecessary clutter in users inboxes.

Here I build a spam email detection model that classifies emails as Spam or Ham (Not Spam) using TensorFlow, one of the most popular deep learning libraries.

####################################################################################################################

Step 1: Importing Required Libraries

Before we begin let’s import the necessary libraries: pandas, numpy, tensorflow, matplotlib, wordcloud, nltk for data processing, model building, and visualization.

I'll create a requirements.txt file with all the necessary dependencies and then install them through the terminal.

TensorFlow: The main deep learning framework for building neural networks
Pandas: For handling and processing email datasets
NumPy: For numerical computations and array operations
Matplotlib: For basic data visualization and plotting
Seaborn: For advanced statistical visualizations
NLTK: For natural language processing tasks like text preprocessing
WordCloud: For creating visual representations of word frequency
Scikit-learn: For machine learning utilities like train-test split and model evaluation

pip install -r requirements.txt

##############################################################################################################

Step 2: Loading the Dataset

I’ll use a dataset containing labeled emails (Spam or Ham). Let’s load the dataset and inspect its structure. 

data = pd.read_csv('Emails.csv')

Now, let's visualize the label distribution to get understanding of the class distribution:

sns.countplot(x='label', data=data)
plt.show()


###############################################################################################################

Step 3: Balance the Dataset

We can clearly see that number of samples of Ham is much more than that of Spam which implies that the dataset we are using is imbalanced. To address the imbalance we’ll downsample the majority class (Ham) to match the minority class (Spam).

ham_msg = data[data['label'] == 'ham']
spam_msg = data[data['label'] == 'spam']

# Downsample Ham emails to match the number of Spam emails
ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)
### Without random_state=42:
        Run 1: Might select emails [1, 5, 12, 23, ...]
        Run 2: Might select emails [3, 8, 15, 27, ...]
        Different results each time!
        With random_state=42:
        Run 1: Always selects emails [1, 5, 12, 23, ...]
        Run 2: Always selects emails [1, 5, 12, 23, ...]
        Same results every time!

# Combine balanced data
balanced_data = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)

whats happening?
pd.concat([ham_msg_balanced, spam_msg])
pd.concat(): Pandas function that combines multiple DataFrames
[ham_msg_balanced, spam_msg]: List of two DataFrames to combine
ham_msg_balanced: The downsampled ham emails (same count as spam)
spam_msg: All the spam emails
Result: Creates a new DataFrame with both datasets stacked vertically
.reset_index(drop=True)
.reset_index(): Resets the DataFrame's index (row numbers)
drop=True: Drops the old index instead of keeping it as a column
Why needed: After concatenation, you might have duplicate index numbers (e.g., both datasets have rows 0, 1, 2, 3...)



# Visualize the balanced dataset
sns.countplot(x='label', data=balanced_data)
plt.title("Balanced Distribution of Spam and Ham Emails")
plt.xticks(ticks=[0, 1], labels=['Ham (Not Spam)', 'Spam'])
plt.show()