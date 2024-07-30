#!/bin/bash

# Check the number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <db_name> <db_user> <db_pass>"
    exit 1
fi

# Database parameters passed through command line arguments
DB_NAME=$1
DB_USER=$2
DB_PASS=$3

# Run psql in terminal to set up the project database
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME;"
sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
sudo -u postgres psql -c "ALTER ROLE $DB_USER SET client_encoding TO 'utf8';"
sudo -u postgres psql -c "ALTER ROLE $DB_USER SET default_transaction_isolation TO 'read committed';"
sudo -u postgres psql -c "ALTER ROLE $DB_USER SET timezone TO 'UTC';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
sudo -u postgres psql -c "ALTER DATABASE $DB_NAME OWNER TO $DB_USER;"

# Update/create .pg_service.conf file in the home directory with created db configuration
PG_SERVICE_CONF="$HOME/.pg_service.conf"
touch $PG_SERVICE_CONF

echo "[food-detect]" >> $PG_SERVICE_CONF
echo "host=localhost" >> $PG_SERVICE_CONF
echo "user=$DB_USER" >> $PG_SERVICE_CONF
echo "dbname=$DB_NAME" >> $PG_SERVICE_CONF
echo "port=5432" >> $PG_SERVICE_CONF
echo "password=$DB_PASS" >> $PG_SERVICE_CONF

# Set file permissions
chmod 600 $PG_SERVICE_CONF

echo "Database and user successfully created, .pg_service.conf file configured."

# Perform Django migrations and populate-db script
cd ./fooddetect
python3 manage.py migrate
python3 manage.py runscript populate-db

echo "Migrations completed and database populated."
