# IRCTC Automation

This project automates the process of booking train tickets on the Indian Railway Catering and Tourism Corporation (IRCTC) website. By utilizing Selenium, it streamlines tasks such as logging in, selecting trains, filling out details, and processing payments.

## Setup

Before running the automation script, ensure you have the following prerequisites:

1. **.env File**: Create a `.env` file in the project root directory. This file should contain the following information:

Username = "IRCTC USER NAME"
Password = "IRCTC PASSWORD"
Upi = "YOUR UPI ID"

    Replace `Username`, `Password`, and `Upi` with your actual IRCTC credentials and UPI ID.

2. **Selenium Driver**: This project utilizes Selenium for web automation. Ensure you have the appropriate Selenium driver for your browser installed and configured. The `Utils` file contains a function to create a Selenium driver.

3. **Model for Captcha**: The `Utils` file also contains functions to read captchas. Make sure you have a model loaded to read captchas effectively.

## Usage

1. **Login**: The first step may involve a one-step or two-step process depending on your browser's settings. If the login process doesn't go smoothly, you may need to adjust this step accordingly.

2. **Select Train**: Once logged in, you can select the train from the list provided, considering the train's name and seat availability.

3. **Payment**: After selecting the desired train and filling out necessary details, the payment amount will be sent to the UPI ID mentioned in the `.env` file. Ensure the UPI ID is correct to receive the payment link.

## Note

- The automation script is designed to handle common scenarios, but slight adjustments may be required based on individual browser configurations.
- Make sure all dependencies are installed and configurations are set up correctly before running the script.

Happy automating your IRCTC ticket booking process! 🚂🎟️
