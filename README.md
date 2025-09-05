Spam messages are unsolicited or unwanted emails/messages sent in bulk to users. Detecting spam emails automatically helps prevent unnecessary clutter in users inboxes.

Here I build a spam email detection model that classifies emails as Spam or Ham (Not Spam) using TensorFlow, one of the most popular deep learning libraries.

## Datasethttps://your-share-link
Download the dataset here: [Emails.csv](https://drive.google.com/file/d/1-PCrAVn3vmIkHImIUGfVW3xaZAqfwHcU/view?usp=sharing)
Place it in the project root.

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

###plt.xticks(ticks=[0, 1], labels=['Ham (Not Spam)', 'Spam'])
        plt.xticks(): Customizes the x-axis labels
        ticks=[0, 1]: Specifies which tick positions to modify (0 and 1 on the x-axis)
        labels=['Ham (Not Spam)', 'Spam']: Replaces the default labels with more descriptive ones
        Why needed: By default, seaborn might show 'ham' and 'spam', but this makes it clearer

plt.show()

#######################################################################################################

Step 4: Clean the Text

Textual data often requires preprocessing before feeding it into a machine learning model. Common steps include removing stopwords, punctuations, and performing stemming/lemmatization.

I performed the following steps:

Stopwords Removal
            Stopwords are very common words like is, am, are, the, a, an, in, on, of, and, but, etc.
            They usually don’t add much meaning to the text and can introduce noise.
            Example:
            Original sentence: "The cat is sitting on the mat"
            After removing stopwords: "cat sitting mat"
            Purpose → Reduce dimensionality and focus on meaningful words.
Punctuations Removal
            Punctuation marks like . , ! ? : ; " ' ( ) don’t usually carry semantic meaning (except in some cases like sentiment analysis with “!”).
            Removing them helps standardize text.
            Example:
            Original: "Wow!!! The movie was awesome, I loved it."
            After punctuation removal: "Wow The movie was awesome I loved it"
Stemming or Lemmatization
            These are techniques to reduce words to their base/root form, so similar words are treated the same.
            Stemming:
            A rule-based, crude approach that chops off word endings.
            Example:
            "playing", "played", "plays" → "play"
            "running" → "run" (sometimes → "runn")
            Lemmatization:
            A smarter, dictionary-based approach that reduces words to their lemma (valid root word).
            Example:
            "playing", "played", "plays" → "play"
            "better" → "good"
            "children" → "child"
            More accurate than stemming, but slower.

Although removing data means loss of information we need to do this to make the data perfect to feed into a machine learning model.

First we will remove the "subject" written in the text coloumn for each row,,,,,,,,,
balanced_data['text'] = balanced_data['text'].str.replace('Subject', '')
####            balanced_data['text']
                Left side: The 'text' column of the balanced_data DataFrame
                Target: This is what we're modifying
                balanced_data['text'].str.replace('Subject', '')
                balanced_data['text']: The 'text' column we're working with
                .str: Accessor that allows string methods on pandas Series
                .replace('Subject', ''): String method that replaces text
                'Subject': The text to find and replace
                '': Empty string - what to replace it with (effectively removes it)
                What it does:
                Removes the word "Subject" from all email text content

print(balanced_data.head())

now we do:

punctuations_list = string.punctuation
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

    **str.maketrans()**: Creates a translation table for character replacement
        **Parameters**: - '' (first): Characters to replace FROM (empty - we're not replacing) -
                          '' (second): Characters to replace TO (empty - we're removing) -
                           punctuations_list (third): Characters to DELETE -
         **Result**: A translation table that maps punctuation to None (deletion)
    ## return text.translate(temp) - **text.translate(temp)**: Applies the translation table to the text - **Effect**: Removes all punctuation characters from the text## 

balanced_data['text']= balanced_data['text'].apply(lambda x: remove_punctuations(x))
print(balanced_data.head())


The below function is a helper function that will help us to remove the stop words.

def remove_stopwords(text):
    stop_words = stopwords.words('english')

    imp_words = []

    # Storing the important words
    for word in str(text).split():
        word = word.lower()

        if word not in stop_words:
            imp_words.append(word)

    output = " ".join(imp_words)

    return output

balanced_data['text'] = balanced_data['text'].apply(lambda text: remove_stopwords(text))
balanced_data.head()

********EXTRA*******************Visualization Word Cloud************************
A word cloud is a text visualization tool that help's us to get insights into the most frequent words present in the corpus of the data.

def plot_word_cloud(data, typ):
    email_corpus = " ".join(data['text'])
    wc = WordCloud(background_color='black', max_words=100, width=800, height=400).generate(email_corpus)
    plt.figure(figsize=(7, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {typ} Emails', fontsize=15)
    plt.axis('off')
    plt.show()

plot_word_cloud(balanced_data[balanced_data['label'] == 'ham'], typ='Non-Spam')
plot_word_cloud(balanced_data[balanced_data['label'] == 'spam'], typ='Spam')

##      Function: plot_word_cloud(data, typ)
        Build corpus: " ".join(data['text']) merges all email texts into one big string.
        Generate word cloud: WordCloud(...).generate(email_corpus) creates a word cloud image from the merged text.
        Plot it:
        plt.figure(...): set figure size.
        plt.imshow(wc, interpolation='bilinear'): display the word cloud.
        plt.title(...): title indicates which class you’re visualizing (ham/spam).
        plt.axis('off'): hides axes.
        plt.show(): renders the plot.
        Calls
        Ham word cloud: plot_word_cloud(balanced_data[balanced_data['label'] == 'ham'], typ='Non-Spam')
        Spam word cloud: plot_word_cloud(balanced_data[balanced_data['label'] == 'spam'], typ='Spam')
        Result: Two visuals showing the most frequent words in non-spam vs spam emails, helping you compare dominant terms per class.


################################################################################################################

Step 6: Tokenization and Padding

Machine learning models work with numbers, so we need to convert the text data into numerical vectors using Tokenization and Padding.

Tokenization: Converts each word into a unique integer.
Padding: Ensures that all text sequences have the same length, making them compatible with the model.

train_X, test_X, train_Y, test_Y = train_test_split(
    balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42
)
        ######## Splits texts and labels into training (80%) and test (20%) sets with a fixed seed.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)
        ######## Builds a word index from training texts only (prevents data leakage).
train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)
        ######## Converts each email into a list of integer word IDs based on the tokenizer’s vocabulary.
max_len = 100  # Maximum sequence length
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    ########## Padding/truncation:
    Ensures all sequences have the same length max_len=100.
    Shorter sequences are padded with zeros at the end; longer ones are cut off at the end.
train_Y = (train_Y == 'spam').astype(int)
test_Y = (test_Y == 'spam').astype(int)
        ###### Converts labels to binary: spam → 1, ham → 0


##############################################################################################################

Step 7: Define the Model

We will build a deep learning model using a Sequential architecture. This model will include:

Embedding Layer: Learns vector representations of words.
LSTM Layer: Captures patterns in sequences.
Fully Connected Layer: Extracts relevant features.
Output Layer: Predicts whether an email is spam or not.

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_len),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])
@###################Embedding: Maps each word id to a 32‑dim vector. Input size = vocab size (len
           (tokenizer.word_index)+1), sequence length = max_len.
            LSTM(16): Learns sequence patterns in the text with 16 hidden units.
            Dense(32, relu): Nonlinear layer to learn higher‑level features.
            Dense(1, sigmoid): Outputs probability of spam (0..1).
            what are the numbers????
            LSTM(16): 16 memory cells/hidden units. More units → can learn more complex sequence patterns, but more parameters and overfitting risk.
            Dense(32): 32 neurons in the fully connected layer. More neurons → higher representational capacity and compute cost.
            You can tune these (e.g., 8/16/32/64) to balance accuracy vs speed/overfitting.
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy']
)
#######     loss: BinaryCrossentropy(from_logits=False) — for 2-class problems; from_logits=False 
            matches your sigmoid output (probabilities 0–1).
            optimizer: 'adam' — adaptive gradient method that usually trains well out of the box.
            metrics: ['accuracy'] — reports classification accuracy during training/validation.

model.build(input_shape=(None, max_len))
            Keras builds models lazily. Before a model sees input, its layers aren’t initialized, so summary() can’t show shapes/params.
            model.build(input_shape=(None, max_len)) tells Keras the expected input shape:
            None: batch size (variable)
            max_len: sequence length (timesteps)
            With the input shape known, Keras initializes layer weights, infers each layer’s output shape, and computes parameter counts. Then model.summary() can display them.
model.summary()

########################################################################################################

Step 8: Train the Model

We train the model using EarlyStopping and ReduceLROnPlateau callbacks. These callbacks help stop the training early if the model’s performance doesn’t improve and reduce the learning rate to fine-tune the model.
EarlyStopping: stops training if val_accuracy doesn’t improve for 3 epochs and restores the best weights.
ReduceLROnPlateau: halves the learning rate if val_loss plateaus for 2 epochs (helps escape plateaus).

es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)

history = model.fit(
    train_sequences, train_Y,
    validation_data=(test_sequences, test_Y),
    epochs=20,
    batch_size=32,
    callbacks=[lr, es]
)

######        model.fit(...): trains for up to 20 epochs on train_sequences/train_Y, validates on 
               test_sequences/test_Y, batch size 32, with the two callbacks.

After training, we evaluate the model on the test data to measure its performance.

test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)

########### model.evaluate(...): computes final loss and accuracy on the test set; you print both.

print('Test Loss :',test_loss)
print('Test Accuracy :',test_accuracy)

        Accuracy: fraction of correct predictions (0–1).
        Loss: value from the loss function (e.g., binary cross-entropy), not bounded to make 1 − accuracy.
        You can have high accuracy with relatively high loss (confident wrong predictions) or low loss with moderate accuracy (well-calibrated probabilities but some errors).

Having trained our model, we can plot a graph depicting the variance of training and validation accuracies with the no. of epochs.

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


########################################################################################################

By following these steps, we have successfully built a machine learning model that can classify emails as spam or ham

IMPORTNAT POINTS :
I used Keras Sequential neural network for binary classification:
Embedding → LSTM(16) → Dense(32, relu) → Dense(1, sigmoid).
Trained with Adam optimizer and Binary Crossentropy loss.


