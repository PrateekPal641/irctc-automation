from selenium import webdriver
import os, random
from selenium.webdriver.common.by import By
import time
import traceback 
import base64
import re
import cv2
import pytesseract
import torch
import onnx
import onnxruntime as rt
from torchvision import transforms as T
from PIL import Image
import pathlib
import gradio as gr

from utils import *


response = {'From':'kanpur', 'To':'lucknow', 'Date':'23', 'Month':'April'}

driver = create_selenium_remote_driver()

availibility = search_trains(driver, response)
print(availibility)
train = input('Enter index of train you want to choose')

do_booking(train=train, driver=driver)
