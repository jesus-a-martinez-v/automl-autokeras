# AutoML Auto-Keras

Simple introductory project to automatic machine learning with Auto-Keras.

## Installation

I recommend you use `virtualenv`. If you have it installed, create a new environment and activate it:

```bash
$> virtualenv -p python3 venv
$> source venv/bin/activate
```

After that, just install the requirements:

```bash
pip install -r requirements.txt
```

## Try It

You can fire up an automatic search like this:

```bash
python train.py -t <max_hours>
```

Where `<max_hours>` is the upper cap to the maximum amount of hours the program will search for a model. Keep in mind 
this is a heavy and long running process. **You will need a GPU**.