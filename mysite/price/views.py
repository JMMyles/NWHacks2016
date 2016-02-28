from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse

from .models import Request

from .forms import RequestForm


def index(request):
	if request.method == 'POST':

		form = RequestForm(request.POST)

		if form.is_valid():

			return HttpResponseRedirect('/results/')

	else:
		form = RequestForm()

	#latest_request_list = Request.objects.order_by('-pub_date')[:5]
	#context = {
		#'latest_request_list': latest_request_list
	#}
	return render(request, 'price/index.html', {'form':form})


def results(request):
	if request.method == 'POST':

		form = RequestForm(request.POST)

		return HttpResponse(form)

	else:

		response = "You're looking at the results of request."
		return HttpResponse(response)

# Create your views here.

