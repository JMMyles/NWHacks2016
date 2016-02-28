# -*- coding: utf-8 -*-
# Generated by Django 1.9.2 on 2016-02-28 09:14
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('price', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='request',
            name='choice_text',
        ),
        migrations.RemoveField(
            model_name='request',
            name='question_text',
        ),
        migrations.AddField(
            model_name='request',
            name='bathroom_text',
            field=models.CharField(default=0, max_length=200),
        ),
        migrations.AddField(
            model_name='request',
            name='bed_text',
            field=models.CharField(default=0, max_length=200),
        ),
        migrations.AddField(
            model_name='request',
            name='size_text',
            field=models.CharField(default=0, max_length=200),
        ),
        migrations.AddField(
            model_name='request',
            name='vip_text',
            field=models.CharField(default=0, max_length=200),
        ),
        migrations.AddField(
            model_name='request',
            name='year_text',
            field=models.CharField(default=0, max_length=200),
        ),
    ]