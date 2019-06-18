#!/bin/bash
set -e


####  Regenerating NeuroML2 from PyNN

cd PyNN

# Tidy up
rm -rf *.xml *.nml


#------------------------------------

python brunel_to_neuroml.py


#------------------------------------

cd ..



