Broad-UNet: Multi-scale feature learning for nowcasting tasks
========

The official code of the following paper : <Link to arXiv>

Results
-----
Some results for the different nowcasting tasks

+------------------------------------------+------------------------------------------------------------------------------------+
|       Task                               | Actual  Vs    Prediction                                                           |
+==========================================+====================================================================================+
| Precipitation prediction (30 mins ahead) |.. figure:: figures/precipitation_pred1.png                                         |          
+------------------------------------------+------------------------------------------------------------------------------------+
| Precipitation prediction (30 mins ahead) |.. figure:: figures/precipitation_pred2.png                                         |
+------------------------------------------+------------------------------------------------------------------------------------+
| Cloud cover prediction (15 mins ahead)   |.. figure:: figures/cloud_pred1.png                                                 |
+------------------------------------------+------------------------------------------------------------------------------------+
| Cloud cover prediction (90 mins ahead)   |.. figure:: figures/cloud_pred1.png                                                 |          
+------------------------------------------+------------------------------------------------------------------------------------+



Installation
-----

The required modules can be installed  via:

.. code:: bash

    pip install -r requirements.txt
    
Quick Start
~~~~~~~~~~~
Depending on the nowcasting task to be performed, the models can be trained running:

.. code:: bash

    python training_clouds_data.py 
    
or 

.. code:: bash

    python training_precipitation_data.py 


To evaluate the models and visualize some predictions, please run:

.. code:: bash

    python training_clouds_data.py 
    
or 

.. code:: bash

    python training_precipitation_data.py 



📜 Scripts
-----

- The scripts contain the models, the generator, the training files and evaluation files.

🔍 Models
-----

We show here the schema related to the AsymmInceptionRes-3DDR-UNet model.

.. figure:: figures/AsymmInceptionRes-3DDR-UNet.png
  
📂 Data
-----

In order to download the data or any of the trained models, please email to the following addresses:

j.garciafernandez@student.maastrichtuniversity.nl

i.alaouiabdellaoui@student.maastrichtuniversity.nl

siamak.mehrkanoon@maastrichtuniversity.nl

The data must be downloaded and unzipped inside the 'Data/' directory.


🔗 Citation
-----

If you use our data and code, please cite the paper using the following bibtex reference:

.. code:: bibtex

    @article{fernandez2020deep,
      title={Deep coastal sea elements forecasting using U-Net based models},
      author={Fern{\'a}ndez, Jes{\'u}s Garc{\'\i}a and Abdellaoui, Ismail Alaoui and Mehrkanoon, Siamak},
      journal={arXiv preprint arXiv:2011.03303},
      year={2020}
    }
