# Generated by Django 5.2.4 on 2025-07-11 11:26

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('search', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='imagedata',
            name='image_embedding',
        ),
    ]
