from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, HttpResponseNotFound, FileResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from . import models
import threading
import time
import os
import zipfile
import tensorflow as tf
import numpy as np
import pickle



node_ids = set()
global_models = [] #paths all global models created
local_models = {} #node id + list of paths local models given
latest_local_models = [] #paths of all latest local models, cleared every round
round_number = 0
current_round_uploads = set()
nodes_deregistered = 0
node_non_participation = 0

def store_file(file, name):
    with open("local_models/" + name, "wb+") as dest:
        for chunk in file.chunks():
            dest.write(chunk)

def start_new_round():
    global latest_local_models, round_number, current_round_uploads, node_non_participation
    latest_local_models = [] 
    round_number += 1
    current_round_uploads = set()
    node_non_participation = 0

def unzip_file(zip_path, extract_to):
    """
    Unzips a ZIP file to the specified directory.

    :param zip_path: Path to the ZIP file.
    :param extract_to: Directory where files should be extracted.
    """
    # Ensure the output directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all the contents into the directory specified
        zip_ref.extractall(extract_to)
        print(f"All files extracted to {extract_to}")

def aggregate_local_models():
    global latest_local_models, round_number, current_round_uploads, node_non_participation
    print("Aggregating local models")

    if len(latest_local_models) > 0:
        models_loaded = []
        
        for model_path in latest_local_models:
            file_name = model_path[:-4]
            saved_path = "local_models_unzipped/" + file_name
            unzip_file("local_models/" + model_path, saved_path)
            
            models_loaded.append(tf.keras.models.load_model(saved_path))
        
        model_weights = [model_loaded.get_weights() for model_loaded in models_loaded]
        # Calculate the average weights across all models
        avg_weights = []
        for weights in zip(*model_weights):
            avg_weights.append(np.mean(weights, axis=0))
        
        weights_saved_path = "global_models/" + "weights_file_" + str(round_number) + '.pkl'
        with open(weights_saved_path, 'wb') as file:
            pickle.dump(avg_weights, file)

        global_models.append(weights_saved_path)

    start_new_round()
    print("Done aggregating")

class StartServerView(View):
    def get(self, request):
        global node_ids, global_models, local_models, latest_local_models, round_number

        node_ids = set()
        global_models = [] #all global models created
        local_models = {} #node id + list of local models given
        latest_local_models = [] #cleared every round
        round_number = 0

        return JsonResponse({"success": True}, safe=False)

class IndexView(View):
    def get(self, request):
        return HttpResponse("Hello, world!")
    

class RegisterView(View):
    def post(self, request):
        form = request.POST
        node_id = form["node_id"]

        global node_ids, local_models
        if node_id not in node_ids:
            node_ids.add(node_id)
            local_models[node_id] = []

            print("New client registered:", node_id)
            
            return JsonResponse({"success": True}, safe=False)
        return JsonResponse({"success": False}, safe=False)
    

class PollingView(View):
    def post(self, request):
        return JsonResponse({"round": round_number}, safe=False)
    
class UploadLocalModelView(View):
    def post(self, request):
        global node_ids, global_models, local_models, latest_local_models, round_number, current_round_uploads, node_non_participation, nodes_deregistered
        
        model_file = request.FILES["model_file"]
        node_id = request.POST["node_id"]
        local_round_number = int(request.POST["round_number"])
        file_format = ".zip" #to be changed
        file_name = str(node_id) + "_" + str(local_round_number) + file_format

        if node_id not in current_round_uploads:
            store_file(model_file, file_name)
            local_models[node_id].append(file_name)
            latest_local_models.append(file_name)
            success = True    
            current_round_uploads.add(node_id)
        else:
            success = False

        #make thread for aggregating latest local models
        if len(latest_local_models) == (len(node_ids) - nodes_deregistered - node_non_participation):
            thread = threading.Thread(target=aggregate_local_models)
            thread.start()

        return JsonResponse({"success": success}, safe=False)
    
class RetrieveLatestModelView(View):
    def post(self, request):
        if round == 0:
            return JsonResponse({"success": False})
        
        latest_model_path = global_models[-1]

        print("Model to be served", latest_model_path)

        if os.path.exists(latest_model_path):
            # Set the content type and headers to prompt download
            response = FileResponse(open(latest_model_path, 'rb'))
            response['Content-Type'] = 'application/octet-stream'
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(latest_model_path)}"'
            return response
        else:
            # Handle file not found error
            return HttpResponse('Sorry. This file is not available.', status=404)
        
class DeregisterView(View):
    def post(self, request):
        global nodes_deregistered
        nodes_deregistered += 1
        print("Node dergistered")

        if len(latest_local_models) == (len(node_ids) - nodes_deregistered - node_non_participation):
            thread = threading.Thread(target=aggregate_local_models)
            thread.start()

        return JsonResponse({"success": True}, safe=False)
    
class NonParticipationView(View):
    def post(self, request):
        global node_non_participation
        node_non_participation += 1
        print("Node not participating")
        print(len(latest_local_models))
        print((len(node_ids) - nodes_deregistered - node_non_participation))

        if len(latest_local_models) == (len(node_ids) - nodes_deregistered - node_non_participation):
            thread = threading.Thread(target=aggregate_local_models)
            thread.start()

        return JsonResponse({"success": True}, safe=False)