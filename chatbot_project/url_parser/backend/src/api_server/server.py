import json
import regex as re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.parser.parser_factory.ParserFactory import ParserFactory
from src.evaluator.TokenEvaluator import TokenEvaluator
from src.evaluator.LengthEvaluator import LengthEvaluator
from src.evaluator.RougeEvaluator import RougeEvaluator
from src.evaluator.BleuEvaluator import BleuEvaluator
from src.exceptions.ParserFactoryException import ParserFactoryException
from src.exceptions.WebParserException import WebParserException
from src.parser.WebParser import WebParser

EXIT_ERROR_CODE: int = 1
URL_REGEX: str = r"^https?:\/\/(?:[\w-]+\.)+[a-zA-Z]{2,}(?:\/[\w\-\.\/?%&=!$'()*+,;]*)?$"
DEBUG: bool = True
class ParseOutput(BaseModel):
    url: str
    domain: str
    title: str
    html_text: str
    parsed_text: str

class SupportedDomains(BaseModel):
    domains: list[str]

class GSEntry(BaseModel):
    url: str
    domain: str
    title: str
    html_text: str
    gold_text: str

class ListGSEntry(BaseModel):
    gold_standard: list[GSEntry]

class RawHTMLParseRequest(BaseModel):
    url: str
    html_text: str

class EvaluationInput(BaseModel):
    parsed_text: str
    gold_text: str

class ParseEvaluation(BaseModel):
    token_level_eval: TokenEvaluator.TokenLevelEval
    length_eval: LengthEvaluator.LengthEval
    rouge_eval: RougeEvaluator.RougeEval
    bleu_eval: BleuEvaluator.BleuEval

gs_data: dict[str, list[dict]] = {}
parse_handler: dict[str, WebParser] | None = None
full_gs_evals: dict[str, ParseEvaluation] = {}

def evaluate_parsing(eval_input: EvaluationInput) -> ParseEvaluation:
    parsed_text: str = eval_input.parsed_text
    gold_text: str = eval_input.gold_text
    token_eval_res: TokenEvaluator.TokenLevelEval = TokenEvaluator().evaluate(gold_text, parsed_text)
    length_eval_res: LengthEvaluator.LengthEval = LengthEvaluator().evaluate(gold_text, parsed_text)
    rouge_eval_res: RougeEvaluator.RougeEval = RougeEvaluator().evaluate(gold_text, parsed_text)
    bleu_eval_res: BleuEvaluator.BleuEval = BleuEvaluator().evaluate(gold_text, parsed_text)
    return ParseEvaluation(token_level_eval=token_eval_res, length_eval= length_eval_res, rouge_eval= rouge_eval_res, bleu_eval= bleu_eval_res)

async def full_gs_eval(domain: str) -> ParseEvaluation:
    if domain not in WebParser.get_supported_domains():
        raise HTTPException(status_code=400, detail="domain not supported")
    
    if full_gs_evals and domain in full_gs_evals:
        return full_gs_evals[domain]
    
    evals: list[ParseEvaluation] = []

    if domain not in gs_data:
        raise HTTPException(status_code=404, detail=f"gold standard not found for domain '{domain}'")

    data: list[dict] = gs_data[domain]

    for entry in data:
        url: str = entry.get("url")
        gold_text: str = entry.get("gold_text")
        try: 
            parse_output: dict[str, str] = await parse_handler[domain].parse_url(url, raw_html=entry.get("html_text"))
        except WebParserException as err:
            if (DEBUG):
                print(f"[API-SERVER] | [ERROR] Failed raw HTML parse for URL '{url}': {repr(err)}")
            raise HTTPException(status_code=500, detail=f"raw HTML parse failed for URL '{url}'")

        if not parse_output:
            raise HTTPException(status_code=500, detail=f"raw HTML parse produced no output for URL '{url}'")

        parsed_text: str = parse_output.get("parsed_text")
        evals.append(evaluate_parsing(EvaluationInput(parsed_text=parsed_text, gold_text=gold_text)))

    if not evals:
        raise HTTPException(status_code=500, detail="unable to retrieve gold standard data")
    
    # extract and divide evals into types
    token_evals: list[TokenEvaluator.TokenLevelEval] = [parse_eval.token_level_eval for parse_eval in evals]
    length_evals: list[LengthEvaluator.LengthEval] = [parse_eval.length_eval for parse_eval in evals]
    rouge_evals: list[RougeEvaluator.RougeEval] = [parse_eval.rouge_eval for parse_eval in evals]
    bleu_evals: list[BleuEvaluator.BleuEval] = [parse_eval.bleu_eval for parse_eval in evals]

    # mean of token_evals
    precisions: list[float] = [e.precision for e in token_evals]
    recalls: list[float] = [e.recall for e in token_evals]
    f1s: list[float] = [e.f1 for e in token_evals]
    full_token_eval: TokenEvaluator.TokenLevelEval = TokenEvaluator.TokenLevelEval(precision= sum(precisions)/len(precisions), recall= sum(recalls)/len(recalls), f1 = sum(f1s)/len(f1s))

    # mean of lenth_evals
    c_ratios: list[float] = [e.char_length_ratio for e in length_evals]
    w_ratios: list[float] = [e.word_length_ratio for e in length_evals]
    full_length_eval: LengthEvaluator.LengthEval = LengthEvaluator.LengthEval(golden_chars=None, parsed_chars=None, golden_words=None, parsed_words=None, char_length_ratio = sum(c_ratios)/len(c_ratios), word_length_ratio = sum(w_ratios)/len(w_ratios))

    # mean of rouge_evals
    r1: list[float] = [e.rouge1_f1 for e in rouge_evals]
    r2: list[float] = [e.rouge2_f1 for e in rouge_evals]
    rL: list[float] = [e.rougeL_f1 for e in rouge_evals]
    full_rouge_eval: RougeEvaluator.RougeEval = RougeEvaluator.RougeEval(rouge1_f1= sum(r1)/len(r1), rouge2_f1= sum(r2)/len(r2), rougeL_f1= sum(rL)/len(rL))

    # mean of bleu_evals
    b1: list[float] = [e.bleu1 for e in bleu_evals]
    b2: list[float] = [e.bleu2 for e in bleu_evals]
    b3: list[float] = [e.bleu3 for e in bleu_evals]
    b4: list[float] = [e.bleu4 for e in bleu_evals]
    bavg: list[float] = [e.bleu_avg for e in bleu_evals]
    full_bleu_eval: BleuEvaluator.BleuEval = BleuEvaluator.BleuEval(bleu1= sum(b1)/len(b1), bleu2= sum(b2)/len(b2), bleu3= sum(b3)/len(b3), bleu4= sum(b4)/len(b4), bleu_avg= sum(bavg)/len(bavg))
    
    return ParseEvaluation(token_level_eval=full_token_eval, length_eval= full_length_eval, \
                           rouge_eval= full_rouge_eval, bleu_eval= full_bleu_eval)

async def preload_full_evals() -> dict[str, ParseEvaluation]:
    evals_dict: dict[str, ParseEvaluation] = {}
    count: int = 1
    if (DEBUG):
        print("[API-SERVER] | [INFO] Commencing full GS evaluation for all domains...")
    for domain in WebParser.get_supported_domains():
        if (DEBUG):
            print(f"[API-SERVER] | [INFO] Fully evaluating domain '{domain}'.")
        evals_dict[domain] = await full_gs_eval(domain)
        if (DEBUG):
            print(f"[API-SERVER] | [INFO] OK. ({count}/{len(WebParser.get_supported_domains())})")
        count += 1
    if (DEBUG):
        print("[API-SERVER] | [INFO] Pre-caching of full evaluations for all GS domains completed successfully.")
    return evals_dict

async def parse_helper(url: str, raw_html: str | None = None) -> dict[str, str]:
    if not (re.match(URL_REGEX, url) and url.count("/") >= 3):
        raise HTTPException(status_code=400, detail="malformed URL")
    
    domain_to_parse: str = url.split("/")[2]
    if (DEBUG):
        print(f"[API-SERVER] | [INFO] Extracted domain from URL: {domain_to_parse}")

    if domain_to_parse not in WebParser.get_supported_domains():
        raise HTTPException(status_code=400, detail="domain not supported")
    
    try:
        parse_output: dict[str, str] = await parse_handler[domain_to_parse].parse_url(url, raw_html=raw_html)
    except WebParserException as err:
        if (DEBUG):
            print(f"[API-SERVER] | [ERROR] Failed to parse URL '{url}': {repr(err)}")
        raise HTTPException(status_code=500, detail="URL parse failed")

    if (len(parse_output) == 0):
        raise HTTPException(status_code=400, detail="unreachable URL")
    
    return parse_output

@asynccontextmanager
async def lifespan(app: FastAPI):
    global gs_data, parse_handler, full_gs_evals
    
    if (DEBUG):
        print("[API-SERVER] | [INFO] Initializing...")

    for domain in WebParser.get_supported_domains():
        file_path: str = "gs_data/" + domain.replace(".", "_") + "_gs.json"
        try:
            with open(file=file_path, mode='r', encoding='UTF-8') as fin:
                gs_data[domain] = json.load(fin)
        except FileNotFoundError as err:
            if (DEBUG):
                print(f"[API-SERVER] | [ERROR] Failed to find path '{file_path}': {repr(err)}")
                print("[API-SERVER] | [FATAL] Unable to complete backend initialization. Shutting down...")
            exit(EXIT_ERROR_CODE)

    if not len(gs_data):
        if (DEBUG):
            print(f"[API-SERVER] | [ERROR] Failed to load GS data.")
            print("[API-SERVER] | [FATAL] Unable to complete backend initialization. Shutting down...")
        exit(EXIT_ERROR_CODE)

    try:
        parse_handler = ParserFactory().get_domain_handlers(gs_data)
    except ParserFactoryException as err:
        if (DEBUG):
            print(f"[API-SERVER] | [ERROR] Failed to initialize domain handlers: {repr(err)}")
            print("[API-SERVER] | [FATAL] Unable to complete backend initialization. Shutting down...")
        exit(EXIT_ERROR_CODE)

    full_gs_evals = await preload_full_evals()

    if (DEBUG):
        print("[API-SERVER] | [INFO] Successfully initialized GS data and required parsers.")
        print("[API-SERVER] | [INFO] Backend initialization complete.")
    
    yield
    
    if (DEBUG):
        print("[API-SERVER] | [INFO] Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/parse")
async def parse_url(url: str) -> ParseOutput:
    if (DEBUG):
        print(f"[API-SERVER] | [INFO] Received parsing request for URL: {url}")
    parse_output: dict[str, str] = await parse_helper(url)
    return ParseOutput(url=parse_output.get("url"), domain=parse_output.get("domain"), title=parse_output.get("title"),
                       html_text=parse_output.get("html_text"), parsed_text=parse_output.get("parsed_text"))

@app.post("/parse")
async def parse_raw_html(req: RawHTMLParseRequest) -> ParseOutput:
    url: str = req.url
    html_text: str = req.html_text
    if (DEBUG):
        print(f"[API-SERVER] | [INFO] Received raw HTML parsing request for URL: {url}")
    parse_output: dict[str, str] = await parse_helper(url, raw_html=html_text)
    return ParseOutput(url=parse_output.get("url"), domain=parse_output.get("domain"), title=parse_output.get("title"),
                       html_text=parse_output.get("html_text"), parsed_text=parse_output.get("parsed_text"))

@app.get("/domains")
def get_supported_domains() -> SupportedDomains:
    return SupportedDomains(domains=WebParser.get_supported_domains())

@app.get("/gold_standard")
def get_gold_standard(url: str) -> GSEntry:
    if not (re.match(URL_REGEX, url) and url.count("/") >= 3):
        raise HTTPException(status_code=400, detail="malformed URL")
    domain: str = url.split("/")[2]
    if domain not in WebParser.get_supported_domains():
        raise HTTPException(status_code=400, detail="domain not supported")
    if domain not in gs_data:
        raise HTTPException(status_code=404, detail="gold standard not found for the given URL")
    for entry in gs_data[domain]:
        if entry.get("url") == url:
            return GSEntry(url=entry.get("url"), domain=entry.get("domain"), title=entry.get("title"),
                           html_text=entry.get("html_text"), gold_text=entry.get("gold_text"))
    raise HTTPException(status_code=404, detail="gold standard not found for the given URL")

@app.get("/full_gold_standard")
def get_all_golden_standard_domain(domain: str) -> ListGSEntry:
    if domain not in WebParser.get_supported_domains():
        raise HTTPException(status_code=400, detail="domain not supported")
    if domain not in gs_data:
        raise HTTPException(status_code=404, detail="gold standard not found for the given URL")
    return ListGSEntry(gold_standard=gs_data[domain])

@app.post("/evaluate")
def evaluate_parsing_endpoint(eval_input: EvaluationInput) -> ParseEvaluation:
    return evaluate_parsing(eval_input)

@app.get("/full_gs_eval")
async def full_gs_eval_endpoint(domain: str) -> ParseEvaluation:
    if domain not in WebParser.get_supported_domains():
        raise HTTPException(status_code=400, detail="domain not supported")
    if full_gs_evals and domain in full_gs_evals:
        return full_gs_evals[domain]
    
    return await full_gs_eval(domain)