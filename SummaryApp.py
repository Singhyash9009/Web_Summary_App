import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by  import By
from selenium.common.exceptions import NoSuchElementException,TimeoutException,WebDriverException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from transformers import BartForConditionalGeneration, BartTokenizer
from urllib.parse import urlsplit
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
import heapq
from time import sleep
import  numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
import warnings
warnings.simplefilter(action='ignore')


@st.cache_data
def file(data):
    scrp=[]
    for i in range(len(website_df['Website_name'])):
        driver = webdriver.Chrome()
        driver.get(f"{website_df['Website_name'][i]}")
        driver.implicitly_wait(10)
        try:
            # Remove header by tag name
            try:
                tag_name=['header','head','h1','h2','h3']
                for tag in tag_name:
                    try:
                        header = driver.find_element(By.TAG_NAME, tag)
                        driver.execute_script('arguments[0].parentNode.removeChild(arguments[0]);', header)
                        break
                    except NoSuchElementException:
                        continue
            except NoSuchElementException as e:
                print(f'Error removing header: {str(e)}')
    
            # Attempt to remove footer using different XPATH expressions
            try:
                footer = driver.find_element(By.TAG_NAME, 'footer')
                driver.execute_script('arguments[0].parentNode.removeChild(arguments[0]);', footer)
            except:
                try:    
                    footer_xpath_list = [
                        '//div[@id="baseFooter"]',
                        '//div[@class="footer-top"]',
                        '//footer[@class="footer"]',
                        '//footer[@id="footer"]',
                        '//footer[contains(@class, "footer")]'
                    ]
                    
                    for xpath in footer_xpath_list:
                        try:
                            footer = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath)))
                            driver.execute_script("arguments[0].parentNode.removeChild(arguments[0]);", footer)
                            break
                        except NoSuchElementException:
                            continue
                except TimeoutException as e:
                    print(f"TimeoutException footer not found: {str(e)}")
                except NoSuchElementException as e:
                    print(f'Error removing footer: {str(e)}')
    
            try:
                button_list = [
                    '//button[@id="onetrust-accept-btn-handler"]',
                    '//a[@id="macs_cookies_accept_necessary"]',
                    '/div[contains(text(),"Accept all")]',
                    '//a[contains(text(),"Accept")]',
                    '//a[contains(text(),"Allow all")]',
                    '//a[contains(text(),"ACCEPT")]',
                    '//button[contains(text(),"Accept all")]',
                    '//span[contains(text(),"Accept all")]'
                ]
                for xpath in button_list:
                    # Accept cookies if the accept button is present
                    try:
                        accept_button = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, xpath)))
                        accept_button.click()
                        break
                    except NoSuchElementException:
                        continue
            except TimeoutException as e:
                print(f"TimeoutException button not found: {str(e)}")
            except NoSuchElementException as e:
                print(f'Cookies Notification not found: {str(e)}')
    
            # Get the text content of the body
            content = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            scrp.append(content.text)
            
            print(20*'*****')
            sleep(3)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
        # Close the WebDriver
        driver.quit()
    website_df['scraped_text']=scrp
    patterns_2_r = [
        '©.*', 'We respect your.*', 'We use cookies.', 'Cookie Policy.*', 'Contact us.*', 'Contact Us.*',
        'Cookie settings.*', 'Cookie policy.*', 'By clicking.*', 'This website uses cookies.','Cookies.*','Cookie Consent.*'
        ]
    
    patterns = '|'.join(patterns_2_r)
    
    # Defined function to clean our text extracted from the website
    def clean_text(text):
        c_text = re.sub(patterns, ' ', text)
        return ''.join(c_text)
    
    website_df['clean_text'] = website_df['scraped_text'].apply(clean_text)
    
    kt = []
    
    for i in range(len(website_df)):
        sentences = website_df['clean_text'][i].split('\n')
    
        tokenizer = Tokenizer()
    
        # Fit the tokenizer on the sentences
        tokenizer.fit_on_texts(sentences)
    
        # Filter sentences based on word count
        filtered_sentences = [sentence for sentence in sentences if len(tokenizer.texts_to_sequences([sentence])[0]) >= 7]
    
        # Reconstruct the text
        kt.append('\n'.join(filtered_sentences))
    
    website_df['clean_text_kt'] = kt
    website_df['clean_text_kt'] = website_df['clean_text_kt'].apply(lambda x: re.sub(r'\s+', ' ', x))
    # website_df.rename(columns={'Unnamed: 0':'Company_name'},inplace=True)
    # Load pre-trained BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    summaries = []
    
    # Generate summaries for each input text
    for i in range(len(website_df)):
        # Tokenize and generate summary
        inputs = tokenizer.encode(website_df['clean_text_kt'][i], return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=250, min_length=80, length_penalty=2.0, num_beams=13, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
        # Append the generated summary to the list
        summaries.append(summary)
    
    website_df['Summarised_text'] = summaries
    
    return website_df,summaries

# st.title("Summary Application Tool")
st.markdown("""
    <div style='background-color: #808080; padding: 10px; border-radius: 5px;'>
        <h1 style='color: white; text-align: center;'>Summary Application Tool</h1>
    </div>
    """,
    unsafe_allow_html=True
    )

# Upload a CSV file with website URLs
uploaded_file = st.file_uploader("Upload a CSV file containing website URLs", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        website_df = pd.read_csv(uploaded_file)

        
        if 'scraped_text' not in website_df.columns:
            st.info("Scraping and summarizing websites. This may take a moment...")
            website_df, summaries = file(website_df)
        else:
            # Summaries have already been generated and cached
            summaries = website_df['Summarised_text'].tolist()
    
       
        current_summary_index = st.session_state.get('current_summary_index', 0)
        st.markdown("""
            <div style='background-color: #808080; padding: 10px; border-radius: 5px;'>
                <h2 style='color: white; text-align: center;'>Summarized Texts</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        # Add Next and Previous buttons
        col1, col2, col3,col4,col5,col6 = st.columns([3,1,7,2,2,1])
        with st.container():
            # # col3.write(f"## Summarized Texts")
            # col3.markdown("""
            #            <div style='background-color: #FFFFFF; padding: 10px; border-radius: 5px;'>
            #                <h2 style='color: black; text-align: center;'>Summarized Texts</h2>
            #            </div>
            #            """,unsafe_allow_html=True)
            col3.markdown(f"""
                       <div style='background-color:  #F7DC6F; padding: 10px; border-radius: 10px;'>
                           <h2 style='color: black; text-align: center;'>{website_df['Website_name'].str.split('.')[current_summary_index][1].capitalize()}</h2>
                       </div>
                       """,unsafe_allow_html=True)
            col3.markdown("<br>", unsafe_allow_html=True)

            col3.markdown(f"""<div class="card border-primary mb-3" style="max-width: 20rem; background-color: #AED6F1;color: black;border-radius: 10px;"><p class="card-text">{summaries[current_summary_index]}</p></div>
        """
        ,unsafe_allow_html=True)
          

        st.markdown(
        """
        <style>
        .previous-button {
            background-color: #8E44AD;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        </style>
        """
        , unsafe_allow_html=True)
  
        col1.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        if col1.button("Previous", key="previous_button") and current_summary_index > 0:
            current_summary_index -= 1
            st.session_state.current_summary_index = current_summary_index
      
        col5.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    
        if col5.button("Next") and current_summary_index < len(summaries) - 1:
            current_summary_index += 1
            st.session_state.current_summary_index = current_summary_index

      
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
