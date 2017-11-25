"""
These classes listen for models being published, execute those models and publish the results to another queue

To start:
source activate tensorflow
from model_executor import *
ModelMessageConsumer().listen()

Make models in 2017-11-23-model-creator. Executing that notebook will publish a
model to the model publishing queue
"""

import json
import datetime
import h5py
import keras
import socket

from kafka import KafkaConsumer
from kafka import KafkaProducer

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.models import model_from_json

from prepare_football_data_for_keras import DataPreperator

class ModelCompletePublisher:
    "Publishes an executed model's results to a message queue"

    results_topic = "test-model-results"

    conf = {
        'bootstrap.servers': ["spark4.thedevranch.net"],
        'session.timeout.ms': 6000,
        'default.topic.config': {'auto.offset.reset': 'smallest'},
    }

    producer = KafkaProducer(bootstrap_servers=conf['bootstrap.servers'], )
    
    def publish_results(self, msg):
        "publishes model results to a kafka queue"

        self.producer.send(self.results_topic,
                           msg.encode('utf-8'))
    
class ModelMessageConsumer:
    "Listens to a message queue for new models being published"

    def listen(self):
        # To consume latest messages and auto-commit offsets
        consumer = KafkaConsumer('test-models',
                                 group_id='test-group-001',
                                 client_id=socket.gethostname()
                                 bootstrap_servers=['spark4.thedevranch.net'],
                                 #value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                                 auto_offset_reset='latest',
                                 enable_auto_commit=False,
                                )
        for message in consumer:
            # message value and key are raw bytes -- decode if necessary!
            # e.g., for unicode: `message.value.decode('utf-8')`
            print("%s:%d:%d: key=%s" % (message.topic, message.partition,
                                                  message.offset, message.key))
            
            ModelExecutor().execute_model(message.value.decode('utf-8'))
            
                                          
class ModelExecutor(ModelCompletePublisher):
    "Given a jsonified model, execute it with the standard data set"
    
    def debug(self, msg):
        print(msg)
        
    def load_data(self):
        self.debug("loading data")
        dp = DataPreperator()
        dp.load_data()
        dp.create_validation_split()
        return dp
        
    def generate_results(self, model, training_hx, training_eval, training_time_start, training_time_end):
        "generate json to write to queue"
        
        evaluation_metrics = dict(zip(model.metrics_names, training_eval))
        evaluation_metrics = {**evaluation_metrics,
                      **training_hx.params,
                      **{'history' : training_hx.history},
                      **{'training_time_start' : training_time_start, 'training_time_end' : training_time_end},
                      **{'model' : json.loads(model.to_json())},
                     }
        
        return evaluation_metrics
    
    def save_model_results(self, model):
        "saves model and weights to disk"
        model_name = "football_cnn-%s.h5" % datetime.datetime.now().isoformat()
        model.save(model_name)
    
    def execute_model(self, json_model):
        
        (train_x,
         train_y,
         test_x,
         test_y,
         val_x,
         val_y) = self.load_data().get_data_sets()
                                          
        model = model_from_json(json_model)
        sgd = keras.optimizers.SGD(lr=1e-3, nesterov=True)

        model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
        
        self.debug(model.summary())
        
        training_time_start = datetime.datetime.now().isoformat()
        training_hx = model.fit(
            train_x,
            train_y,
            batch_size=128, #220
            epochs=20, #FIXME: turn this back into 10
            verbose=1,
            validation_data=(val_x, val_y))
        
        training_time_end = datetime.datetime.now().isoformat()
        
        training_eval = model.evaluate(test_x, test_y, batch_size=128, verbose=1)
        
        self.debug(dict(zip(model.metrics_names, training_eval)))
        
        evaluation_metrics = self.generate_results(model,
                                                   training_hx,
                                                   training_eval,
                                                   training_time_start,
                                                   training_time_end)
        
        #self.debug(json.dumps(evaluation_metrics, indent=2))
        self.publish_results(json.dumps(evaluation_metrics))
        self.save_model_results(model)
        