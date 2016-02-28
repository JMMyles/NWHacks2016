from django.db import models

# Create your models here.

class Request(models.Model):
	vip_text = models.CharField( max_length=200)
	year_text = models.CharField( max_length=200)
	bathroom_text = models.CharField( max_length=200)
	size_text = models.CharField( max_length=200)
	bed_text = models.CharField(max_length=200)
	pub_date = models.DateTimeField('date published')

	def __str__(self):
		return self.year_text + " " + self.bathroom_text + " " + self.size_text + " " + self.bed_text + " " + self.vip_text










