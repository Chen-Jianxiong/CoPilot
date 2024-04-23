import json
import logging
import time
import uuid
from base64 import b64decode
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.cors import CORSMiddleware

from app import routers
from app.config import PATH_PREFIX, PRODUCTION
from app.log import req_id_cv
from app.metrics.prometheus_metrics import metrics as pmetrics
from app.tools.logwriter import LogWriter

# 创建FastAPI实例，如果是生产环境，则禁用文档
if PRODUCTION:
    app = FastAPI(
        title="TigerGraph CoPilot", docs_url=None, redoc_url=None, openapi_url=None
    )
else:
    app = FastAPI(title="TigerGraph CoPilot")

# 配置FastAPI框架的跨域资源共享（CORS）中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置路由，统一指定前缀为 环境变量"PATH_PREFIX"
app.include_router(routers.root_router, prefix=PATH_PREFIX)
app.include_router(routers.inquiryai_router, prefix=PATH_PREFIX)
app.include_router(routers.supportai_router, prefix=PATH_PREFIX)

# 这些路径将不会被收集到Metrics中
excluded_metrics_paths = ("/docs", "/openapi.json", "/metrics")

# 定义了一个日志对象，用于记录应用中的日志信息。__name__是一个特殊变量，表示当前模块的名称。
logger = logging.getLogger(__name__)


async def get_basic_auth_credentials(request: Request):
    """异步，从HTTP请求中提取 basic auth 用户名和密码。"""
    auth_header = request.headers.get("Authorization")

    if auth_header is None:
        return ""

    # 将该字段拆分为"Basic"和Base64编码的凭据。
    try:
        auth_type, encoded_credentials = auth_header.split(" ", 1)
    except ValueError:
        return ""

    if auth_type.lower() != "basic":
        return ""

    try:
        # 将Base64编码的凭据解码为UTF-8格式的字符串，并提取用户名和密码。
        decoded_credentials = b64decode(encoded_credentials).decode("utf-8")
        username, _ = decoded_credentials.split(":", 1)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return username


# 定义一个HTTP中间件（log_requests）
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """异步，记录每个请求的日志信息。属于一个名为log_requests的中间件（middleware）
    当一个HTTP请求进入时，它会生成一个请求ID（req_id），记录进入日志，并将请求ID存储在req_id_cv变量中。
    然后，它调用下一个中间件或路由处理函数，处理请求。
    如果请求失败，将状态设置为"FAILURE"。如果请求成功，将状态设置为"SUCCESS"。"""

    req_id = str(uuid.uuid4())
    LogWriter.info(f"{request.url.path} ENTRY request_id={req_id}")
    # 请求上下文变量
    req_id_cv.set(req_id)
    start_time = time.time()
    # 调用下一个中间件或路由处理函数，处理请求。
    response = await call_next(request)

    # 从HTTP请求中提取 basic auth 用户名和密码。
    user_name = await get_basic_auth_credentials(request)
    client_host = request.client.host
    # 用户代理
    user_agent = request.headers.get("user-agent", "Unknown")
    action_name = request.url.path
    status = "SUCCESS"

    if response.status_code != 200:
        status = "FAILURE"

    # 设置审计日志条目结构，并使用 LogWriter 编写它
    if not any(request.url.path.endswith(path) for path in excluded_metrics_paths):
        audit_log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "userName": user_name,
            "clientHost": f"{client_host}:{request.url.port}",
            "userAgent": user_agent,
            "endpoint": request.url.path,
            "actionName": action_name,
            "status": status,
            "requestId": req_id,
        }
        LogWriter.audit_log(json.dumps(audit_log_entry), mask_pii=False)
        update_metrics(start_time=start_time, label=request.url.path)

    return response


def update_metrics(start_time, label):
    duration = time.time() - start_time
    # 发送一个计数器，表示在 duration 时间戳内，名为copilot_endpoint_duration_seconds的指标下，观察到label
    pmetrics.copilot_endpoint_duration_seconds.labels(label).observe(duration)
    # inc：递增计数器的值。
    pmetrics.copilot_endpoint_total.labels(label).inc()
