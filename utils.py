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
from tokenizer_base import Tokenizer
import pathlib
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

Username = os.environ.get("Username")
Password = os.environ.get("Password")
Upi = os.environ.get("Upi")

def create_selenium_remote_driver(url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(30)
    driver.set_page_load_timeout(60)
    driver.set_window_size(1512, 860)
    driver.get(url=url)
    return driver
def get_transform():
        img_size = (32,128)
        transforms = []
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        return T.Compose(transforms)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def initialize_model(model_file):
    transform = get_transform()
    # Onnx model loading
    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)
    ort_session = rt.InferenceSession(model_file)
    return transform,ort_session 

def get_text(image_path, transform,ort_session ):
    img_org = Image.open(image_path)
    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
    x = transform(img_org.convert('RGB')).unsqueeze(0)
    charset = r"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    tokenizer_base = Tokenizer(charset)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    logits = ort_session.run(None, ort_inputs)[0]
    probs = torch.tensor(logits).softmax(-1)
    preds, probs = tokenizer_base.decode(probs)
    print(preds, probs)
    return preds[0], probs
def decode_captcha(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    print(data)
    return data
def extract_text(driver):
    src = driver.find_element(By.XPATH, "//img[@class='captcha-img']").get_attribute("src")
    regex_pattern = r'data:image\/[a-zA-Z]+;base64,([a-zA-Z0-9+/=]+)'
    matches = re.search(regex_pattern, src)
    if matches:
        base64_data = matches.group(1)
        image_b64 = base64_data

    imgdata = base64.b64decode(image_b64)
    filename = 'last_captcha.png'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    with open('captchas/'+str(random.randint(1, 1000))+'.png', 'wb') as f:
        f.write(imgdata)

    try:
        captcha_data=decode_captcha(filename)
        captcha_text=''.join(captcha_data['text'])
        captcha_text= captcha_text.replace(' ', '')
        confidence = [conf for conf in captcha_data['conf'] if conf!=-1]
        confidence = min(confidence)
    except:
        transform,ort_session = initialize_model(model_file='captcha.onnx')
        captcha_text, probs = get_text(filename, transform=transform, ort_session=ort_session)
        confidence=min(probs)

    return (captcha_text, confidence)

def search_trains(driver, response):
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[1]/app-header/div[1]/div[2]/a/i").click()
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[1]/app-header/div[3]/p-sidebar/div/nav/div/label/button").click()
    ###Username
    driver.find_element(By.XPATH, "//input[@placeholder='User Name']").send_keys(Username)
    ###Password
    driver.find_element(By.XPATH, "//input[@placeholder='Password']").send_keys(Password)
    ###Captcha
    confidence=0
    start=0
    while confidence<90:
        if start ==0:
            captcha_text, confidence = extract_text(driver)
            start=1
        else:
            driver.find_element(By.XPATH, "//span[@class='glyphicon glyphicon-repeat']").click()
            captcha_text, confidence = extract_text(driver)
    print(captcha_text, confidence)

    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/app-login/p-dialog[1]/div/div/div[2]/div[2]/div/div[2]/div[2]/div[2]/form/div[4]/div/app-captcha/div/div/input").send_keys(captcha_text)

    driver.implicitly_wait(0)
    ##Click Sign In
    try:
        driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/app-login/p-dialog[1]/div/div/div[2]/div[2]/div/div[2]/div[2]/div[2]/form/span/button").click()
    except:
        pass
    driver.implicitly_wait(30)



    ##FROM

    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[1]/div[1]/p-autocomplete/span/input").clear()
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[1]/div[1]/p-autocomplete/span/input").send_keys(response['From'])
    try:
        driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[1]/div[1]/p-autocomplete/span/div/ul/li[1]").click()
    except:
        driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[1]/div[1]/p-autocomplete/span/div/ul/li").click()

    ###TO
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[1]/div[2]/p-autocomplete/span/input").clear()
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[1]/div[2]/p-autocomplete/span/input").send_keys(response['To'])
    try:
        driver.find_element(By.XPATH,"/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[1]/div[2]/p-autocomplete/span/div/ul/li").click()
    except:
        driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[1]/div[2]/p-autocomplete/span/div/ul/li[1]").click()


    
    ##Month
    
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[2]/div[1]/p-calendar/span/input").click()
    time.sleep(1)
    current_month = driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[2]/div[1]/p-calendar/span/div/div/div[1]/div/span[1]").text
    count=0
    print('before starting...........',str(current_month), response['Month'])
    while str(current_month).strip()!=response['Month'].strip() and count<4:
        ###Next Month
        print(str(current_month), response['Month'])
        driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[2]/div[1]/p-calendar/span/div/div/div[1]/a[2]/span").click()
        current_month = driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[2]/div[1]/p-calendar/span/div/div/div[1]/div/span[1]").text
        count+=1
    if count==4:
        return "Invalid Month"

    ###Date
    date_clicked = False
    driver.implicitly_wait(0)
    for i in range(1, 6):
        if date_clicked==True:
            break

        for j in range(1,8):
            xpath = "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[2]/div[2]/div[1]/p-calendar/span/div/div/div[2]/table/tbody/tr["+str(i)+"]/td["+str(j)+"]/a"
            try:
                element = driver.find_element(By.XPATH,xpath)
                print(str(element.text), response['Date'])
                if response['Date'].strip() == str(element.text).strip():
                    print(xpath)
                    
                    driver.find_element(By.XPATH,xpath).click()
                    date_clicked=True
                    break
            except:
                pass
    driver.implicitly_wait(30)
    
    driver.implicitly_wait(1)
    try:
        driver.find_element(By.XPATH, "//span[contains(text(), 'Yes')]")
    except:
        pass
    driver.implicitly_wait(30)




    ###Search
    if date_clicked==True:
        driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-main-page/div/div/div[1]/div[2]/div[1]/app-jp-input/div/form/div[5]/div[1]/button").click()
    else:
        return "Invalid Date"





    all_trains = driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-train-list/div[4]/div/div[1]/span").text.strip()
    all_trains = int(all_trains.split(" ")[0])
    if all_trains>10:
        all_trains= all_trains//2
    availibility = []

  
    for i in range (1,all_trains+1):
        try:
            refresh_element = "/html/body/app-root/app-home/div[3]/div/app-train-list/div[4]/div/div[5]/div["+str(i)+"]/div[1]/app-train-avl-enq/div[1]/div[5]/div[1]/table/tr/td[1]/div/div[2]"
            if driver.find_element(By.XPATH, refresh_element).text=="Refresh":
                driver.find_element(By.XPATH, refresh_element).click()

            seat_availability=driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-train-list/div[4]/div/div[5]/div["+str(i)+"]/div[1]/app-train-avl-enq/div[1]/div[7]/div[1]/div[3]/table/tr/td[2]/div/div[2]/strong").text
            train_name=driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-train-list/div[4]/div/div[5]/div["+str(i)+"]/div[1]/app-train-avl-enq/div[1]/div[1]/div[1]/strong").text
            availibility.append({train_name:seat_availability})
        except:
            pass
    return availibility

def do_booking(driver, train):

    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-train-list/div[4]/div/div[5]/div["+str(train)+"]/div[1]/app-train-avl-enq/div[1]/div[7]/div[1]/div[3]/table/tr/td[2]/div/div[2]/strong").click()
    time.sleep(1)
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-train-list/div[4]/div/div[5]/div["+str(train)+"]/div[1]/app-train-avl-enq/div[2]/div/span/span[1]/button").click()

    driver.implicitly_wait(0)
    try:
        driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-train-list/p-confirmdialog[2]/div/div/div[3]/button[1]/span[2]").click()
    except:
        pass
    driver.implicitly_wait(30)

    driver.find_element(By.XPATH, "//input[@placeholder='Passenger Name']").send_keys('Passenger Name')
    driver.find_element(By.XPATH, "//input[@placeholder='Age']").send_keys(22)
    driver.find_element(By.XPATH, "//option[contains(text(),'Male')]").click()
    driver.implicitly_wait(0)
    try:
        driver.find_element(By.XPATH, "//option[contains(text(),'Veg')]").click()
    except:
        pass
    driver.implicitly_wait(60)


    ###Pay Through UPI
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-passenger-input/div[5]/form/div/div[1]/div[12]/p-panel/div/div[2]/div/table/tr[2]/label/p-radiobutton/div/div[2]/span").click()
    ###Continue
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-passenger-input/div[5]/form/div/div[1]/div[14]/div/button[2]").click()
    ###Verify Information
    time.sleep(1)
    confidence=0
    start=0
    while confidence<90:
        if start ==0:
            captcha_text, confidence = extract_text(driver)
            start=1
        else:
            driver.find_element(By.XPATH, "//span[@class='glyphicon glyphicon-repeat']").click()
            captcha_text, confidence = extract_text(driver)
    print(captcha_text, confidence)

    driver.find_element(By.XPATH, "//input[@placeholder='Enter Captcha']").send_keys(captcha_text)
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-review-booking/div[4]/div/div[1]/form/div[3]/div/button[2]").click()

    ###Choose Payment Method
    time.sleep(1)
    driver.find_element(By.XPATH, "/html/body/app-root/app-home/div[3]/div/app-payment-options/div[4]/div[2]/div[1]/div[1]/app-payment/div[2]/button[2]").click()


    ###Ask For Payment
    driver.find_element(By.XPATH, "/html/body/div[6]/div[2]/div[3]/div[1]/div[1]/input").send_keys(Upi)

    # driver.find_element(By.XPATH, "/html/body/div[6]/div[2]/div[3]/div[1]/div[3]/input[3]").click()
    time.sleep(4)
    driver.quit()
    return "success"