FROM php:8.1-fpm

# Install system dependencies
RUN apt-get update && apt-get install -y --fix-missing \
    build-essential \
    libpng-dev \
    libjpeg62-turbo-dev \
    libfreetype6-dev \
    locales \
    zip \
    jpegoptim optipng pngquant gifsicle \
    vim \
    unzip \
    git \
    curl \
    libonig-dev \
    pkg-config \
    libzip-dev

# Clear cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PHP extensions
RUN docker-php-ext-configure gd --with-freetype --with-jpeg && \
    docker-php-ext-install -j$(nproc) gd pdo_mysql mbstring zip exif pcntl

# Install Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# Set working directory
WORKDIR /var/www

# Copy project files
COPY . /var/www

# Copy scripts
COPY wait-for-it.sh /usr/bin/wait-for-it.sh
COPY start.sh /usr/local/bin/start.sh

# Ensure scripts are executable
RUN chmod +x /usr/local/bin/start.sh /usr/bin/wait-for-it.sh

# Install project dependencies
RUN composer install

# Set the entrypoint to the start script
ENTRYPOINT ["start.sh"]
