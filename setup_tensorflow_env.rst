************************
Setup Tensorflow for R&D
************************

#. Install tensorflow

    - install tensorflow using pip: ``pip install --upgrade tensorflow``
        - `Install tensorflow <https://tensorflow.google.cn/install>`_
        - `install tensorflow on apple m1 machine <https://developer.apple.com/metal/tensorflow-plugin/>`_

    - Install using docker (*recommended*)

        .. code-block:: bash

            docker pull tensorflow/tensorflow:latest  # Download latest stable image
            docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter  # Start Jupyter server
            # install tensorflow-serving container for macbook with m1 chip: https://github.com/tensorflow/serving/issues/1816

    - Install tensorflow-datasets
        - ``pip install tensorflow-datasets``: The stable version, released every few months.
        - ``pip install tfds-nightly``: Released every day, contains the last versions of the datasets.

#. start tfx container

    .. code-block:: bash

        # fetch tfx image
        docker pull tensorflow/tfx

        # if you encounter 'No id provided.' error when running tfx container, you may need specify `--entrypoint` option
        docker run -it --mount type=bind,src=$(pwd),dst=/workspace --entrypoint bash tensorflow/tfx
        #docker run -p 33243:6006 -ti --entrypoint bash --mount type=bind,src=/opt/home/cuiyongbo/docker-scaffold,dst=/workspace 0fbc116a552e

        cd && mkdir .keras && cd .keras/ && ln -fs /workspace/datasets/ datasets

        # attach to a running container
        docker container exec -it bffd65ffbadb bash

        # install tensorflow-doc
        pip3 install git+https://github.com/tensorflow/docs

#. python3 to start tensorboard: ``python3 -m tensorboard.main --logdir=/path/to/logs``

#. Supress tensorflow warnings

    .. code-block:: py

        # in scripts
        import os
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # 0 = all messages are logged (default behavior)
        # 1 = INFO messages are not printed
        # 2 = INFO and WARNING messages are not printed
        # 3 = INFO, WARNING, and ERROR messages are not printed

    .. code-block:: sh

        # in bash add environment variable
        export TF_CPP_MIN_LOG_LEVEL=2


.. rubric:: Footnotes

.. [#] `tensorflow/playground <https://github.com/tensorflow/playground>`_
