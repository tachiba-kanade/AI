# Generated by Django 5.2.4 on 2025-07-11 11:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('search', '0002_remove_imagedata_image_embedding'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagedata',
            name='image',
            field=models.FileField(upload_to='image_data/'),
        ),
    ]
