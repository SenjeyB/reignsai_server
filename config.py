import os


class Config:
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', 'memory://')
    RATELIMIT_DEFAULT = "100 per hour"
    POOL_SIZE = int(os.getenv('POOL_SIZE', 100))
    MAX_CARDS_PER_REQUEST = int(os.getenv('MAX_CARDS_PER_REQUEST', 20))


class ProductionConfig(Config):
    DEBUG = False
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')


class DevelopmentConfig(Config):
    DEBUG = True


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
