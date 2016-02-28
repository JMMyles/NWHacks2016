from django import forms
from django.forms import TextInput 

class RequestForm(forms.Form):
    House_size = forms.CharField(label='House Size', max_length=100,widget=forms.TextInput(attrs={'class':'number'}))
    number_of_beds = forms.CharField(label='Number of Beds', max_length=100,widget=forms.TextInput(attrs={'class':'number'}))
    number_of_bathrooms = forms.CharField(label='Number of Bathrooms', max_length=100,widget=forms.TextInput(attrs={'class':'number'}))
    month = forms.CharField(label='Month', max_length=100,widget=forms.TextInput(attrs={'class':'number'}))
    year = forms.CharField(initial = 2016, label='Year', max_length=100,widget=forms.TextInput(attrs={'class':'number'}))

