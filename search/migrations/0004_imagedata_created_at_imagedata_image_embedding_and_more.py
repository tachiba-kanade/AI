# Generated by Django 5.2.4 on 2025-07-11 15:41

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('search', '0003_alter_imagedata_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagedata',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='imagedata',
            name='image_embedding',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='imagedata',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
        ),
    ]
