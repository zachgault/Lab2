import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os

def Intro():
    st.header("Satellite Oribit simulator")
#    image_path = os.path.join(os.path.dirname(__file__), 'Images', '')
#    st.image(image_path, width=400)
    st.write("This webpage accesses real data from in oribit satellites to provie a visual display of their oribits")
    st.write("---")
Intro()


'''
provide some basic facts about satellites like how many their are and the importance of being able to predict where they will be in the future (space junk being a problem)
'''

title = st.text_input("Enter TLE value you want to search for", '0')

def sat_facts():
    st.write("")

def importance():
    st.write("")




    
