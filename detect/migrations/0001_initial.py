# Generated by Django 3.0.3 on 2020-03-02 08:53

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Helmet',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('helmet_Main_Img', models.ImageField(upload_to='images/')),
            ],
            options={
                'db_table': 'helmet',
            },
        ),
    ]