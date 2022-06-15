import copy
import seaborn as sns
from seaborn.utils import os, np, plt, pd
from sklearn import preprocessing as s_prep, metrics as s_mtr, feature_selection as s_fs, pipeline as s_pipe
from tensorflow.keras import models, layers, backend as K, callbacks


class SentimentAnalysis:


    @staticmethod
    def show_layer_shapes(nn_model):
        for i in range(len(nn_model.layers)):

            print(f'Layer {i}: \nInput_shape: {nn_model.layers[i].input_shape}' +
                 f'\nOutput shape: {nn_model.layers[i].output_shape}\n\n')
    
    @staticmethod
    def get_maximum_len(train_encoded, test_encoded):
        return  max([max(list(map(len, train_encoded))), max(list(map(len, test_encoded)))])

    @staticmethod
    def convert_to_vocab(encoded_data, int_to_vocab):
        all_sentences = []
        for sentence in encoded_data:
            all_sentences.append([int_to_vocab[word_int] for word_int in sentence])
        return all_sentences

    @staticmethod
    def convert_to_integer(data, vocab_to_int):
        all_sentences = []
        for sentence in data:
    #         print(sentence)
            all_sentences.append([vocab_to_int[word] if word in vocab_to_int else vocab_to_int["<UNK>"] for word in sentence.split()])
        return all_sentences

    @staticmethod
    def report_with_conf_matrix(y_true, pred):
        """print classification report and return the corresponding
        confusion matrix.
        Return: confusion_matrix_plot"""
        
        print(s_mtr.classification_report(y_true, pred))
        
        sns.set_style('white')
        ax1 = s_mtr.ConfusionMatrixDisplay.from_predictions(y_true, pred)
        plt.title("Confusion Matrix", weight='bold')
        return ax1

    @staticmethod
    def compute_balanced_weights(y: 'array', as_samp_weights=True):
        """compute balanced sample weights for unbalanced classes.
        idea is from sklearn:
        balanced_weight = total_samples / (no of classes * count_per_class)
        WHERE:
        total_samples = len(y)
        no of classes = len(np.unique(y))
        no of samples per class = pd.Series(y).value_counts().sort_index().values
        unique_weights = no of classes * no of samples per class
        samp_weights = {labe:weight for labe, weight in zip(np.unique(y), unique_weights)}
        
        weight_per_labe = np.vectorize(lambda l, weight_dict: weight_dict[l])(y, samp_weights)
        Return:
        weight_per_labe: if samp_weights is True
        class_weights: if samp_weights is False"""
        
        y = np.array(y)
        n_samples = len(y)
        n_classes = len(np.unique(y))
        samples_per_class = pd.Series(y).value_counts().sort_index(ascending=True).values
        denom = samples_per_class * n_classes
        unique_weights = n_samples/denom
        cls_weights = {l:w for l, w in zip(np.unique(y), unique_weights)}
        
        if as_samp_weights:
            return np.vectorize(lambda l, weight_dict: weight_dict[l])(y, cls_weights)
        return cls_weights

    @staticmethod
    def give_percentage(arr: pd.Series):
        """output the percentage of each element in array"""
        
        ser = pd.Series(arr)
        return np.round(100*ser/ser.sum(), 2)
        
    @staticmethod
    def remove_puncs_lcase(sentence):
        sw1 = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        return ''.join([str.lower(char) for char in sentence if char not in sw1])
        
    @staticmethod
    def text_cleaner(text_body: str, remove_stop_words=False, added_stopwords=None):
        """remove punctuation or [and stop words] from text_body.
        Return:
        sentence: str"""
        
        import string
        
        stwrds = s_fex.text.ENGLISH_STOP_WORDS + set(added_stopwords)
        
        nopunc = [char.lower() for char in str(text_body) if char not in string.punctuation]
        nopunc_sentence = ''.join(nopunc)
        if remove_stop_words:
            return ' '.join([word for word in str(nopunc_sentence).split() if word.lower() not in stwrds])
        return nopunc_sentence
        
    @staticmethod
    def nn_weights_biases(model_instance: 'Keras_model'):
        """
        get weights and biases of a neural network
        :param model_instance: 
        :return: weights, biases
        """
        print("Weights and biases are given below:\n"+
              f"{model_instance.weights}")
        params = model_instance.get_weights()
        weights = [params[i] for i in range(len(params)) if i % 2 == 0]
        biases = [params[i] for i in range(len(params)) if i % 2 != 0]
        return weights, biases
        
    @staticmethod
    def update_vocab(old_vocab: dict, new_sentences: list):
        """update a BOW vocabulary with new terms from new_sentences
        old_vocab: old bow vocabulary
        new_sentences: list of new sentences (str)
        Return: updated_dict"""
        
        def check_keys(old_vocab, token):
            return str.lower(token) in old_vocab.keys()
        
        cvect = s_fex.text.CountVectorizer(ngram_range=(1, 3)).fit(new_sentences)
        new_vocab = cvect.vocabulary_
        vocab = dict(old_vocab)
        last_count = max(old_vocab.values())
        for w in new_vocab.keys():
            if check_keys(old_vocab, w):
                continue
            last_count += 1
            vocab[w] = last_count
        return vocab
        
    @staticmethod
    def get_part_of_speech(sentence: str):
        """get root words of words in a sentence.
        Returns
        tagged_words: list of tuples of word, part of speech pairs"""
        
        # split up sentences into components
        sentence_components = word_tokenize(sentence)
        # get part of speech for each word
        tagged_words = pos_tag(sentence_components, tagset='universal')
        return tagged_words
    
    @staticmethod
    def get_filepaths(base_folder: str, search_in_fname: str=None):
        """compile filepaths and return a list of fullpaths"""

        for fpath, folders, filenames in os.walk(base_folder):
            full_names = []
            for fname in sorted(filenames):
                if search_in_fname:
                    if str.lower(search_in_fname) in str.lower(fname):
                        fullpath = f"{fpath}//{fname}"
                        full_names.append(fullpath)
                else:
                    fullpath = f"{fpath}//{fname}"
                    full_names.append(fullpath)
        return full_names

    @staticmethod
    def read_file(fname):
        with open(fname, encoding='utf8', errors='ignore') as f:
            return f.readlines()

    @staticmethod
    def read_many_files(fpaths: list):

        cls = SentimentAnalysis()

        n_files = len(fpaths)
        docs = []
        for n in range(n_files):
            txt = cls.read_file(fpaths[n])
            df = pd.DataFrame(txt)
            review = df[0].apply(lambda x: x.split('\t')[0])
            sentiment = df[0].apply(lambda x: x.split('\t')[1])
            df = pd.concat([review, sentiment], 
                          axis=1)
            cols = ['review', 'sentiment']
            df.columns = cols
            docs.append(df)

        return pd.concat(docs).reset_index(drop=True)
        
    @staticmethod
    def plot_scatter(x, y, condition_on=None, plot_title='Scatter Plot', title_size=14, marker=None, color=None, 
                         paletter='viridis', x_labe=None, y_labe=None, xy_labe_size=8, axis=None, figsize=(8, 4), dpi=200,
                         rotate_xticklabe=False, alpha=None, savefig=False, fig_filename='scatterplot.png'):
            """plot scatter graph on an axis.
            Return: axis """
            
            cls = SentimentAnalysis()
            
            if not axis:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                axis = sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker, palette=paletter, color=color)
            else:
                sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker,
                                alpha=alpha, ax=axis, palette=paletter, color=color)
            axis.set_title(plot_title, weight='bold', size=title_size)
            if x_labe:
                axis.set_xlabel(str.capitalize(x_labe), weight='bold', size=xy_labe_size)
            if y_labe:
                axis.set_ylabel(str.capitalize(y_labe), weight='bold', size=xy_labe_size)
                
            if savefig:
                print(cls.fig_writer(fig_filename, fig))
            return axis
        
    @staticmethod
    def plot_line(x, y, condition_on=None, plot_title='Line Plot', line_size=None, legend_labe=None, show_legend=False,
                marker=None, color=None,  x_labe=None, y_labe=None, axis=None, figsize=(8, 4), dpi=200,
                savefig=False, fig_filename='lineplot.png'):
        """plot line graph on an axis.
        Return: axis """
        
        cls = SentimentAnalysis()
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, size=line_size, label=legend_labe, 
                               legend=show_legend, palette='viridis', color=color)
        else:
            sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, size=line_size, legend=show_legend,
                                  ax=axis, palette='viridis', color=color, label=legend_labe)
        
        axis.set_title(plot_title, weight='bold')
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold')
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold')
        
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
                          
    @staticmethod
    def plot_column(x, y, condition_on=None, plot_title='A Column Chart', x_labe=None, y_labe=None, annotate=True,
                color=None, paletter='viridis', conf_intvl=None, include_perc=False, xy_labe_size=8,
                annot_size=6, top_labe_gap=None, h_labe_shift=0.1, top_labe_color='black', bot_labe_color='blue', 
                index_order: bool=True, rotate_xticklabe=False, title_size=15, xlim: tuple=None, ylim: tuple=None, axis=None, 
                figsize=(8, 4), dpi=200, savefig=False, fig_filename='columnplot.png'):
        """plot bar graph on an axis.
        If include_perc is True, then perc_freq must be provided.
        :Return: axis """
        
        cls = SentimentAnalysis()
        freq_col = pd.Series(y)
        total = freq_col.sum()
        
        if color:
            paletter = None
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.barplot(x=x, y=y, hue=condition_on, 
                              palette=paletter, color=color, ci=conf_intvl)
        else:
            sns.barplot(x=x, y=y, hue=condition_on, ci=conf_intvl, 
                        palette=paletter, color=color, ax=axis)
        
        axis.set_title(plot_title, weight='bold', size=title_size)
        
        if rotate_xticklabe:
                axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold', size=xy_labe_size)
        
        y_range = y.max() - y.min()
        if not top_labe_gap:
            top_labe_gap=y_range/1000
        
        if annotate: 
            cont = axis.containers
            for i in range(len(cont)):
#                 print(len(cont))
                axis.bar_label(axis.containers[i], color=bot_labe_color, size=annot_size,
                                  weight='bold')
                    
                if include_perc:
                    
                    for n, p in enumerate(axis.patches):
                        x, y = p.get_xy()
                        i = p.get_height()
                        perc = round(100 * (i/total), 2)
                        labe = f'{perc}%'
                        top_labe_pos = i+top_labe_gap
                        axis.text(x-h_labe_shift, top_labe_pos, labe, color=top_labe_color, 
                                  size=annot_size, weight='bold')
        
        if xlim:
            axis.set_xlim(left=xlim[0], right=xlim[1])
        
        if ylim:
            axis.set_ylim(bottom=ylim[0], top=ylim[1])
            
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
        
    @staticmethod
    def plot_bar(x, y, condition_on=None, plot_title='A Bar Chart', title_size=15, x_labe=None, y_labe=None,
             color=None, paletter='viridis', conf_intvl=None, include_perc=False, perc_freq=None, annot_size=6,
             bot_labe_gap=10, top_labe_gap=10, v_labe_shift=0.4, top_labe_color='black', xy_labe_size=8,
             bot_labe_color='blue', index_order: bool=True, rotate_yticklabe=False, annotate=False, axis=None,
             xlim=None, figsize=(8, 4), dpi=150, savefig=False, fig_filename='barplot.png'):
        """plot bar graph on an axis.
        If include_perc is True, then perc_freq must be provided.
        :Return: axis """
        
        cls = SentimentAnalysis()
        freq_col = x

        if color:
            paletter = None

        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.barplot(x=x, y=y, hue=condition_on, orient='h',
                              palette=paletter, color=color, ci=conf_intvl)
        else:
            sns.barplot(x=x, y=y, hue=condition_on, ci=conf_intvl, 
                        palette=paletter, color=color, orient='h', ax=axis)

        axis.set_title(plot_title, weight='bold', size=title_size,)

        if rotate_yticklabe:
            axis.set_yticklabels(axis.get_yticklabels(), rotation=90)
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold', size=xy_labe_size)
        if xlim:
            axis.set_xlim(left=xlim[0], right=xlim[1])
        if annotate: 
            cont = axis.containers
#                 print(len(cont))
            for i in range(len(cont)):
                axis.bar_label(axis.containers[i], color=bot_labe_color, size=annot_size,
                weight='bold')
                if include_perc and perc_freq is not None:
                    labe = f'({perc_freq.iloc[i]}%)'
                    axis.text(freq_col.iloc[i]+top_labe_gap, i-v_labe_shift, labe,
                              color=top_labe_color, size=annot_size, weight='bold')
                              
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
    
    @staticmethod
    def plot_box(x=None, y=None, condition_on=None, plot_title="A Boxplot", orientation='horizontal', 
                 x_labe=None, y_labe=None, axis=None, figsize=(8, 4), dpi=150, savefig=False, fig_filename='boxplot.png'):
        """A box distribution plot on an axis.
        Return: axis"""
        
        cls = SentimentAnalysis()
        
        if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
            raise TypeError("x must be a pandas series or numpy array")
        elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
            raise TypeError("y must be a pandas series or numpy array")
            
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.boxplot(x=x, y=y, hue=condition_on, orient=orientation)
        else:
            sns.boxplot(x=x, y=y, hue=condition_on, orient=orientation, ax=axis)
        axis.set_title(plot_title, weight='bold')
        if x_labe:
                axis.set_xlabel(str.capitalize(x_labe), weight='bold')
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold')
            
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
    
    @staticmethod
    def plot_freq(data: 'DataFrame or Series', freq_col_name: str=None,  plot_title: str='Frequency Plot', 
                  include_perc=False, annot_size=6, top_labe_gap=None, bot_labe_gap=None, h_labe_shift=0.4, 
                  index_order: bool=True, fig_h: int=6, fig_w=4, dpi=150, x_labe=None, y_labe=None, color=None, 
                  rotate_xticklabe=False, axis=None, savefig=False, fig_filename='freqplot.png'):
        """plot bar chart on an axis using a frequecy table
        :Return: axis"""
        
        cls = SentimentAnalysis()
        
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Data must be a dataframe or series")
            
        if isinstance(data, pd.Series):
            freq_col = data.value_counts().sort_index()
        
        elif isinstance(data, pd.DataFrame):
            freq_col = data[freq_col_name].value_counts().sort_index()

        paletter = 'viridis'
        if color:
            paletter = None
        
        if not index_order:
            freq_col = freq_col.sort_values()
            
        if include_perc:
            perc_freq = np.round(100 * freq_col/len(data), 2)
        
        if not axis:
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            axis = sns.barplot(x=freq_col.index, y=freq_col, palette=paletter, color=color)
        else:
            sns.barplot(x=freq_col.index, y=freq_col, palette=paletter, color=color, ax=axis)
        
        y_range = freq_col.max() - freq_col.min()
        if not bot_labe_gap:
            bot_labe_gap=y_range/1000
        if not top_labe_gap:
            top_labe_gap=y_range/30
            
        for i in range(len(freq_col)):
            labe = freq_col.iloc[i]
            axis.text(i-h_labe_shift, freq_col.iloc[i]+bot_labe_gap, labe,
                   size=annot_size, weight='bold')
            if include_perc:
                labe = f'({perc_freq.iloc[i]}%)'
                axis.text(i-h_labe_shift, freq_col.iloc[i]+top_labe_gap, labe,
                       color='blue', size=annot_size, weight='bold')
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold')
        else:
            axis.set_xlabel(str.capitalize(freq_col.name))
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold') 
        else:
            axis.set_ylabel('Count')
        
        if rotate_xticklabe:
            axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
        axis.set_title(plot_title, weight='bold', size=15, x=0.5, y=1.025)
        
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis 