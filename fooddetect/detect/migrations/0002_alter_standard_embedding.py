# Generated by Django 5.0.7 on 2024-07-21 19:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detect', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='standard',
            name='embedding',
            field=models.TextField(blank=True),
        ),
    ]
