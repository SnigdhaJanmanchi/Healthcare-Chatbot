# Healthcare-Chatbot

This project implements an AI-driven customer support chatbot tailored for healthcare queries. It uses zero-shot intent classification combined with semantic similarity to respond to common user questions. The chatbot is deployed on Hugging Face Spaces using Gradio.

## Project Overview

The chatbot is capable of understanding and responding to queries about:

* Appointment scheduling
* Clinic operating hours
* Insurance-related inquiries
* Test results
* Prescription refills
* Telehealth availability
* General support and services

It classifies the user's intent and retrieves the most appropriate answer from a custom-built FAQ dataset, without requiring task-specific model training.

## Technologies Used

| Component        | Technology                                   |
| ---------------- | -------------------------------------------- |
| Intent Detection | `transformers` (`facebook/bart-large-mnli`)  |
| Semantic Search  | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| User Interface   | `Gradio`                                     |
| Development      | `Google Colab`                               |
| Deployment       | `Hugging Face Spaces`                        |

## Files Included

| File                             | Description                                                    |
| -------------------------------- | -------------------------------------------------------------- |
| `app.py`                         | Main script containing the chatbot logic and UI                |
| `requirements.txt`               | List of dependencies for building the Hugging Face Space       |
| `healthcare_chatbot_colab.ipynb` | Colab notebook used to prepare and export the deployment files |


## How to Run on Hugging Face Spaces

To deploy the chatbot:

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Select:

   * **Space SDK**: Gradio
   * **Python version**: 3.10 or above
4. Upload the following files:

   * `app.py`
   * `requirements.txt`
5. Wait for the build and deployment to complete. This typically takes 1–2 minutes.
6. Once ready, you will see a public URL to access your chatbot.

## Live Demo

You can test the chatbot at the following link:
[Healthcare Chatbot on Hugging Face](https://huggingface.co/spaces/Sjanmanchi/Healthcare_Chatbot)

