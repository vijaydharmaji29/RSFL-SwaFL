from typing import Dict, Text
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from tensorflow.keras import Sequential, layers, Model
import os
import requests
import time
import pickle
import sys
import csv


from DataManager import DataManager
import Training

import os
import zipfile
import time
import psutil
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
tf.get_logger().setLevel('ERROR')

base_url = "http://20.11.65.172:8000/"

cpu_ram_usage = []
round_start_end_time = []
round_participation = []
scores = []
running_flag = False
global_model_rmse = []


def measure_cpu_ram():
    global cpu_ram_usage, running_flag

    while running_flag == True:
        time_now = time.time()
        cpu_usage = psutil.cpu_percent(interval=1)
        ram_usage = psutil.virtual_memory().percent
        cpu_ram_usage.append((time_now, cpu_usage, ram_usage))
        time.sleep(1)


def zip_directory(folder_path, output_zip_path):
    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # os.walk() generates the file names in a directory tree
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create the full path to the file
                file_path = os.path.join(root, file)
                # Create the archive name, which is the path within the zip file
                # This is the path relative to the folder being zipped
                archive_name = os.path.relpath(file_path, start=folder_path)
                # Add the file to the zip file with its relative archive name
                zipf.write(file_path, arcname=archive_name)

def run(_name, _numberOfRounds = 40, _threshold = 0):
    global round_participation, round_start_end_time, scores
    thread = threading.Thread(target=measure_cpu_ram)
    thread.start()

    print(f"Staring node {_name} - rounds {_numberOfRounds} - threshold {_threshold}")

    dataManager = DataManager(10000)
    numberOfRounds = _numberOfRounds
    name = _name
    threshold = _threshold
    accuracy_distribution = [1]*10

    trainingManager = Training.TrainingManager(dataManager)

    #register client
    register_data = {'node_id': name}
    register_url = base_url + "/aggregator/register/"
    response = requests.post(register_url, data=register_data)
    print(response.text)

    prev_round_start_time = 0
    prev_round_end_time = 1
    checkRoundNumber = 0


    for roundNumber in range(numberOfRounds):
        round_start_time = time.time()
        print("Round Number:", roundNumber)

        # prepare training and testing data
        skewedRatings, cluster_sizes = dataManager.generateSkewedDataset(10)

        #checking for participation
        bhattacharya_distance = dataManager.calculateBhattacharyaDistance(cluster_sizes, accuracy_distribution)
        system_score = (prev_round_end_time - prev_round_start_time)

        print("Bhattacharya Distance:", bhattacharya_distance)
        print("System Score:", system_score)

        overall_score = bhattacharya_distance*10000/system_score
        print("Overall Score:", overall_score)

        result = overall_score > threshold

        scores.append((bhattacharya_distance, system_score, overall_score))

        if result or (roundNumber == 0):
            round_participation.append(True)
            print("Participating in this round")
            trainingManager.train(skewedRatings)
            
            #saving the trained model
            trainingManager.model.save("./saved_model/")
            zip_directory("./saved_model/", "./saved_model_" + name + ".zip")

            #send the zip file
            # URL to the server that will receive the file
            upload_url = base_url + '/aggregator/upload_model/'

            # Path to the file you want to send
            file_path = "./saved_model_" + name + ".zip"

            # Open the file in binary mode
            with open(file_path, 'rb') as f:
                # Define the name of the form field (as expected by the server) and the file to upload
                files = {'model_file': (file_path, f)}

                # Optional: any additional data you want to send in the form
                data = {'node_id': name, 'round_number': roundNumber}

                # Send the POST request
                response = requests.post(upload_url, files=files, data=data)

                # Check the response from the server
                print(response.text)

        else:
            round_participation.append(False)
            non_participate_url = base_url + '/aggregator/non_participate/'
            response = requests.post(non_participate_url)
            print(response)
            print("Client Not Participating")

        #polling and then downloading the latest model
        while True:
            #send post request
            #get latest model number
            polling_url = base_url + "/aggregator/polling/"
            response = requests.post(polling_url)
            response = response.json()
            if int(response["round"]) > checkRoundNumber:
                checkRoundNumber = int(response["round"])
                break

            time.sleep(5)
            print("Polling")


        #downloading the latest model
        url_download = base_url + '/aggregator/download_model/'

        # Send the POST request
        response = requests.post(url_download)
        success = False
        # Check if the request was successful
        try:
            # Save the file
            with open('global_weights.pkl', 'wb') as f:
                f.write(response.content)
            print("File downloaded successfully!")
            success = True
        except:
            print("Failed to retrieve the file.")

        if success:
            with open('global_weights.pkl', 'rb') as file:
                global_weights_list = pickle.load(file)

            trainingManager.model.set_weights(global_weights_list)
            accuracy_distribution, evaluation = trainingManager.testAccuracy()

            rms = evaluation['root_mean_squared_error']
            loss = evaluation['loss']

            global_model_rmse.append((rms, loss))

            print("Evaluation:", evaluation)
            print("Accuracy Distribution:", accuracy_distribution)
        
        prev_round_end_time = time.time()
        prev_round_start_time = round_start_time
    
        round_start_end_time.append((prev_round_start_time, prev_round_end_time))

        write_analysis()
        print("Round complete.")
    
    print("Execution Complete")
    deregister_url = base_url + "/aggregator/deregister_node/"
    response = requests.post(deregister_url)
    print("Deregistered node")

def write_analysis():
    with open("./analysis/round_start_end_time.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        # Write all rows at once
        writer.writerows(round_start_end_time)


    with open("./analysis/round_participation.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        # Write all rows at once
        new_round_participation = [(x,) for x in round_participation]
        writer.writerows(new_round_participation)

    with open("./analysis/cpu_ram_usage.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        # Write all rows at once
        writer.writerows(cpu_ram_usage)

    with open("./analysis/scores.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        # Write all rows at once
        writer.writerows(scores)

    with open("./analysis/rmse.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        # Write all rows at once
        writer.writerows(global_model_rmse)
    
    return

if __name__ == "__main__":
    n = len(sys.argv)
    if n <= 1:
        print("Need to submit command line arguments")
        print("<script> <name> <number of rounds> <threshold>")
    elif n <= 3:
        running_flag = True
        run(sys.argv[1])
    else:
        running_flag = True
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    running_flag = False

    write_analysis()
        
