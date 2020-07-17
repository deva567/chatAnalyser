import streamlit as st
import pandas as pd
import os
import re
import regex
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from os import path
import emoji
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def startsWithDateAndTime(s):
    pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9][0-9]) (AM|PM|am|pm)? -'
    result = re.match(pattern, s)
    if result:
        return True
    return False

def FindAuthor(s):
    patterns = [
        '([\w]+):',                        # First Name
        '([\w]+[\s]+[\w]+):',              # First Name + Last Name
        '([\w]+[\s]+[\w]+[\s]+[\w]+):',    # First Name + Middle Name + Last Name
        '([+]\d{2} \d{5} \d{5}):',         # Mobile Number (India)
        '([+]\d{2} \d{3} \d{3} \d{4}):',   # Mobile Number (US)
        '([\w]+)[\u263a-\U0001f999]+:',    # Name and Emoji              
    ]
    pattern = '^' + '|'.join(patterns)
    result = re.match(pattern, s)
    if result:
        return True
    return False

def getDataPoint(line):   
    splitLine = line.split(' - ') 
    dateTime = splitLine[0]
    date, time = dateTime.split(', ') 
    message = ' '.join(splitLine[1:])
    if FindAuthor(message): 
        splitMessage = message.split(': ') 
        author = splitMessage[0] 
        message = ' '.join(splitMessage[1:])
    else:
        author = None
    return date, time, author, message
def split_count(text):

    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list
def transform_format(val):
    if val == 0:
        return val
    else:
        return 255


def main():
    """Semi Automated ML App with Streamlit """
    activities = ["ChatAnalyzer"]	
    choice = st.sidebar.selectbox("Select Activities",activities)

    if choice == 'ChatAnalyzer':
        result = st.file_uploader("Upload whats app chat Txt file ", type="txt")

        try:
            if result is not None:
                res=result.getvalue()
                pattern ='([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9][0-9]) (AM|PM|am|pm)? -'
                tpl=[(m.start(0), m.end(0)) for m in re.finditer(pattern, res)]
                lst_tpl=[(i[0]) for i in tpl]
                final_op=[(res[lst_tpl[i]:lst_tpl[i+1]]) for i in range(len(lst_tpl)-1)]

                st.subheader("Exploratory Data Analysis")
                date_lst=[]
                time_lst=[]
                author_lst=[]
                msg_lst=[]
                for fp in final_op[2:]:
                    # fp.readline() # Skipping first line of the file because contains information related to something about end-to-end encryption
                   
                    # date, time, author = None, None, None
                    
                    line = fp
                    # st.write(line)

                    if startsWithDateAndTime(line): 
                        date, time, author, message = getDataPoint(line) 
                        date_lst.append(date)
                        time_lst.append(time)
                        author_lst.append(author)
                        msg_lst.append(message)
    
                df = pd.DataFrame(zip(date_lst,time_lst,author_lst,msg_lst), columns=['Date', 'Time', 'Author', 'Message']) # Initialising a pandas Dataframe.
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.dropna()
                URLPATTERN = r'(https?://\S+)'
                df['urlcount'] = df.Message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
                st.dataframe(df.head())
                if st.checkbox("Show Selected Author"):
                    media_messages_df = df[df['Message'] == '<Media omitted>']
                    messages_df = df.drop(media_messages_df.index)
                    messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
                    messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
                    messages_df["MessageCount"]=1
                    df["emoji"] = df["Message"].apply(split_count)
                    messages_df["emojicount"]= df['emoji'].str.len()

                    all_authors=messages_df.Author.unique()
                    selected_columns = st.multiselect("Select only one Author",all_authors)
                    # print(selected_columns[0])
                    req_df= messages_df[messages_df["Author"] == selected_columns[0]]
                    # st.write(req_df)
                    # req_df will contain messages of only one particular user
                    st.write(f'Stats of {selected_columns[0]} -')
                    # shape will print number of rows which indirectly means the number of messages
                    st.write('Messages Sent', req_df.shape[0])
                    #Word_Count contains of total words in one message. Sum of all words/ Total Messages will yield words per message
                    words_per_message = (np.sum(req_df['Word_Count']))/req_df.shape[0]
                    st.write('Words per message', words_per_message)
                    #media conists of media messages
                    media = media_messages_df[media_messages_df['Author'] == selected_columns[0]].shape[0]
                    st.write('Media Messages Sent', media)
                    # emojis conists of total emojis
                    emojis = sum(req_df['emojicount'])
                    print('Emojis Sent', emojis)
                    #links consist of total links
                    links = sum(req_df["urlcount"])   
                    st.write('Links Sent', links)  
                    wine_mask = np.array(Image.open(r"C:\Users\venna\OneDrive\Desktop\bird.png"))
                    transformed_wine_mask = np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)
                    for i in range(len(wine_mask)):
                        transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))
                    dummy_df = messages_df[messages_df['Author'] == selected_columns[0]]
                    text = " ".join(review for review in dummy_df.Message)
                    stopwords = set(STOPWORDS)
                    stopwords.update(["ha","ok","lo","ni","Ha","Ok","Ni","OK","anna","ra", "ga", "na", "ani", "em", "ki", "ah","ha","anta","kuda","ante","la","eh","Nen","ne","haa","Haa","le"])
                    # Generate a word cloud image
                    print('Author name',selected_columns[0])
                    wordcloud = WordCloud(max_font_size=50, max_words=100,stopwords=stopwords,mask=transformed_wine_mask,contour_width=3, contour_color='firebrick', background_color="white").generate(text)
                    # Display the generated image:
                    # the matplotlib way:
                    
                    plt.figure( figsize=[40,20])
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    plt.show() 
                    st.pyplot()
            
                    # st.dataframe(messages_df)


                    # total_emojis_list = list(set([b for b in messages_df['emojicount']]))
                    # total_emojis = len(total_emojis_list)
                    # emoji_dict = dict(Counter(total_emojis_list))
                    # emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
                    # st.write(emoji_dict)
                    # dummy_df = messages_df[messages_df['Author'] == selected_columns[0]]
                    # total_emojis_list = list([b for b in dummy_df.emojicount])
                    # emoji_dict = dict(Counter(total_emojis_list)) 
                    # emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
                    # st.write('Emoji Distribution for',selected_columns[0] )
                    # author_emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])
                    # st.write(author_emoji_df)
                    # fig1, ax1 = plt.subplots()
                    # ax1.pie(author_emoji_df.count, labels=author_emoji_df.emoji, autopct='%1.1f%%', shadow=True, startangle=90)
                    # pie_plot = author_emoji_df.emoji.value_counts().plot.pie(autopct="%1.1f%%")
                    # st.write( ax1.pie)
                    # st.pyplot()
                    # fig = px.pie(author_emoji_df, values='count', names='emoji')
                    # fig.update_traces(textposition='inside', textinfo='percent+label')
                    # fig.show()
        except FileNotFoundError:
            st.error('File not found.')
        except IndexError:
            st.error('Select only one Author. ')

        


if __name__ == '__main__':
	main()
