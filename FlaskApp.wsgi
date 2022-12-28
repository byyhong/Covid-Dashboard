#!/usr/bin/python3.8
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/FlaskApp/")

from FlaskApp import server as application
application.secret_key = 'dfsdlfkjsldkfjaoifjljf'
