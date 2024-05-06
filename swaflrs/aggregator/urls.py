from django.urls import path, include
from . import views

urlpatterns = [
    path("", views.IndexView.as_view()),
    path("register/", views.RegisterView.as_view()),
    path("upload_model/", views.UploadLocalModelView.as_view()),
    path("polling/", views.PollingView.as_view()),
    path("download_model/", views.RetrieveLatestModelView.as_view()),
    path("deregister_node/", views.DeregisterView.as_view()),
    path("non_participate/", views.NonParticipationView.as_view())
]
