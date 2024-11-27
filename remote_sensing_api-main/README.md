
#!/bin/bash

# Create the Virtual Environment
export WORKON_HOME=~/.virtualenvs
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
mkvirtualenv --python=/usr/bin/python3.10 GIS_env

# Activate the virtual environment
source ~/.virtualenvs/GIS_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# GDAL Installation
# Install GDAL development libraries
sudo apt-get install libgdal-dev

# Export environment variables for compiler
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# Install Python GDAL bindings
pip install GDAL==3.4.1

# Update Django settings (Optional)
if [ "$(uname -s)" == "Darwin" ]; then
    VENV_BASE=$(echo $VIRTUAL_ENV | sed 's:/bin/activate::')
    export PATH=$VENV_BASE/lib/python3.10/site-packages/osgeo:$PATH
    export PROJ_LIB=$VENV_BASE/lib/python3.10/site-packages/osgeo/data/proj
fi

# PostgreSQL Installation with PostGIS
# Update system
sudo apt update

# Install PostgreSQL and dependencies
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql

# Create PostgreSQL role and database
sudo -i -u postgres psql -c "CREATE USER mappers WITH PASSWORD 'mappers123';"
sudo -i -u postgres psql -c "CREATE DATABASE digitalmap;"

# Install PostGIS
sudo apt-get install postgresql-16-postgis-3

# Connect to database and install extension
sudo -i -u postgres psql -d digitalmap -c "GRANT ALL PRIVILEGES ON SCHEMA public TO mappers;"
sudo -i -u postgres psql -d digitalmap -c "CREATE EXTENSION postgis;"

# Grant ownership to user (Optional)
sudo -i -u postgres psql -d digitalmap -c "ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO mappers;"

# Make user a superuser (Optional)
sudo -i -u postgres psql -c "ALTER USER mappers WITH SUPERUSER;"

echo "Setup complete!"
