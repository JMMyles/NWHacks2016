from django import forms
from django.forms import TextInput 

class RequestForm(forms.Form):
    house_size = forms.CharField(label='house_size', max_length=100,widget=forms.TextInput(attrs={'type':'number'}))
    number_of_beds = forms.CharField(label='number_of_beds', max_length=100)
    number_of_bathrooms = forms.CharField(label='number_of_bathrooms', max_length=100)
    house_zip = forms.CharField(label='house_zip', max_length=100)
    year = forms.CharField(initial = 2016, label='year', max_length=100)

