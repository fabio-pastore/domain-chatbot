import requests
from requests import Response

# headers needed because wikipedia gets mad about this but not user agent being minervachatbotbot 1.0
def post_data(req_url: str,json_payload: dict | None, headers: dict[str, str] | None = None) -> dict[str, dict|int|bool|str]:
    """
    Posts data to the specified URL with the given headers and JSON payload.
    Args:
        req_url (str): The URL to which the data should be posted.
        json_payload (dict | None): The JSON payload to send with the request.
        headers (dict[str, str]): The headers to include in the request. Default is None.

    Returns:
        dict[str, dict|int|bool|str]: A dictionary containing the response data, status code, and response status.
    """
    response: Response | None = None
    try:
        response = requests.post(req_url, headers=headers, json=json_payload)
        
        try:
            response_data = response.json()
        except:
            response_data = {"detail": response.text if response else "unknown error"}
        
        if not response.ok:
            err_detail = response_data.get("detail", f"HTTP {response.status_code}")
            return {"response_data": {"detail": err_detail}, "status_code": response.status_code, "response_ok": False}
        
        return {"response_data": response_data, "status_code": response.status_code, "response_ok": response.ok}

    except requests.exceptions.RequestException as err:
        print(f"[ERROR] API call error: {err}")
        err_detail: str | None = None
        if response:
            try:
                err_detail = response.json().get("detail")
            except:
                pass
        return {"response_data": {"detail": err_detail if err_detail else str(err)}, "status_code": response.status_code if response else None, "response_ok": False}

def get_data(req_url: str, params: dict[str, str], headers: dict[str, str] | None = None) -> dict[str, dict|int|bool|str]:
    """
    Gets data from the specified URL with the given headers and parameters.

    Args:
        req_url (str): The URL from which the data should be retrieved.
        params (dict[str, str]): The parameters to include in the request.
        headers (dict[str, str]): The headers to include in the request. Default is None.

    Returns:
        dict[str, dict|int|bool|str]: A dictionary containing the response data, status code, and response status.
    """
    response: Response | None = None
    try:
        response = requests.get(req_url, headers=headers, params=params)

        try:
            response_data = response.json()
        except:
            response_data = {"detail": response.text if response else "unknown error"}

        if not response.ok:
            err_detail = response_data.get("detail", f"HTTP {response.status_code}")
            return {"response_data": {"detail": err_detail}, "status_code": response.status_code, "response_ok": False}

        return {"response_data": response_data, "status_code": response.status_code, "response_ok": True}

    except requests.exceptions.RequestException as err:
        print(f"[ERROR] API call error: {err}")
        err_detail: str | None = None
        if response:
            try:
                err_detail = response.json().get("detail")
            except:
                pass
        return {"response_data": {"detail": err_detail if err_detail else str(err)}, "status_code": response.status_code if response else None, "response_ok": False}
