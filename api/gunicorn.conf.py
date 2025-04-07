wsgi_app = "app.api.main:app"

workers = 4

worker_class = "uvicorn.workers.UvicornWorker"

bind = "0.0.0.0:8000"

accesslog = "-"
errorlog = "-"
loglevel = "info"
