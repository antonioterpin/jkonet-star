Installation guide
==================

To start, clone the repo:

.. code-block:: bash

   git clone git@github.com:antonioterpin/jkonet-star.git

.. tabs::

   .. tab:: Docker

      Before proceeding, ensure Docker is installed on your machine. You can download Docker from the official site: `https://www.docker.com/ <https://www.docker.com/>`_.

      Once Docker is installed and running, follow these steps to build the Docker image. Execute the following command from the root directory of the repository:

      .. code-block:: bash

          docker build -t jkonet-star-app .

      If you encounter any issues with the Docker build, please ensure that Docker is running and that you have the necessary permissions to execute Docker commands. You can also try to pull the ``python:3.12-slim`` image before building the ``jkonet-star-app`` image:

      .. code-block:: bash

          docker pull python:3.12-slim

      **Running JKOnet\* using Docker**

      After building the image, you can generate data and train models by executing the following commands:

      .. code-block:: bash

         # Generate population data
         docker run -v "$(pwd)/:/app" jkonet-star-app python data_generator.py --potential wavy_plateau --dataset-name test-wavy-plateau

         # Train the model on the generated dataset
         docker run -v "$(pwd)/:/app" jkonet-star-app python train.py --solver jkonet-star-potential --dataset test-wavy-plateau

   .. tab:: MacOS

      These steps have been tested on MacOS 13.2.1 and should also work on Ubuntu systems.

      Steps:

      1. **Install Miniconda**

         Download and install Miniconda from the official website: `https://docs.conda.io/en/latest/miniconda.html <https://docs.conda.io/en/latest/miniconda.html>`_.

      2. **Create a Conda environment**

         Open a terminal and run the following commands to create and activate a new Conda environment:

         .. code-block:: bash

             conda create --name jkonet-star python=3.12
             conda activate jkonet-star

      3. **Install the required packages**

         Once the environment is activated, install the necessary dependencies:

         .. code-block:: bash

             pip install -r requirements.txt

         To install ``parallel`` (used for running the benchmarks), you can use the following command on MacOS:

         .. code-block:: bash

             brew install parallel

      4. **Test the installation**

         You can generate data and train models by executing the following commands:

         .. code-block:: bash

             # Generate population data
             python data_generator.py --potential wavy_plateau --dataset-name test-wavy-plateau

             # Train the model on the generated dataset
             python train.py --solver jkonet-star-potential --dataset test-wavy-plateau

   .. tab:: Ubuntu

      These steps have been tested on Ubuntu systems.

      Steps:

      1. **Install Miniconda**

         Download and install Miniconda from the official website: `https://docs.conda.io/en/latest/miniconda.html <https://docs.conda.io/en/latest/miniconda.html>`_.

      2. **Create a Conda environment**

         Open a terminal and run the following commands to create and activate a new Conda environment:

         .. code-block:: bash

             conda create --name jkonet-star python=3.12
             conda activate jkonet-star

      3. **Install the required packages**

         Once the environment is activated, install the necessary dependencies:

         .. code-block:: bash

             pip install -r requirements.txt

         To install ``parallel`` (used for running the benchmarks), you can use the following command on Ubuntu:

         .. code-block:: bash

             sudo apt-get install parallel

      4. **Test the installation**

         You can generate data and train models by executing the following commands:

         .. code-block:: bash

             # Generate population data
             python data_generator.py --potential wavy_plateau --dataset-name test-wavy-plateau

             # Train the model on the generated dataset
             python train.py --solver jkonet-star-potential --dataset test-wavy-plateau

   .. tab:: Windows

      The following instructions are for Windows 11 users. Please note that Python 3.9 is required for compatibility.

      Steps:

      1. **Install Miniconda**

         Download and install Miniconda from the official website: `https://docs.conda.io/en/latest/miniconda.html <https://docs.conda.io/en/latest/miniconda.html>`_.

      2. **Create a Conda environment**

         Run the following commands in your terminal to create and activate the environment with Python 3.9:

         .. code-block:: bash

             conda create --name jkonet-star python=3.9
             conda activate jkonet-star

      3. **Install the required packages**

         Once the environment is activated, install the necessary dependencies for Windows:

         .. code-block:: bash

             pip install -r requirements-win.txt

      4. **Test the installation**

         You can generate data and train models by executing the following commands:

         .. code-block:: bash

             # Generate population data
             python data_generator.py --potential wavy_plateau --dataset-name test-wavy-plateau

             # Train the model on the generated dataset
             python train.py --solver jkonet-star-potential --dataset test-wavy-plateau

         .. note::
            Due to maximum filename length limitations, please use the ``dataset-name`` argument. Using the automatically generated filenames might result in errors on Windows.

.. note::
   The installation instructions we provide are not GPU friendly. If you have a GPU, you can install the necessary packages for GPU support. Running the experiments on a GPU yields significant speedups, especially for the JKOnet\* full model (``jkonet-star``). We collected the training times on a RTX 4090.