#!/bin/bash

# Wait for MySQL to be ready
/usr/bin/wait-for-it.sh db:3306 --timeout=60 --strict

# Run database migrations
php artisan migrate --force

# Run seeders
php artisan db:seed --force

# Start PHP-FPM
php-fpm
