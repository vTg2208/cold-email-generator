import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv('GROQ_API_KEY'), model_name='llama-3.3-70b-versatile')

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE
            {page_data}
            ### INSTRUCTION:
            The scraped text is from career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys:
            'role', 'experience', 'skills' and 'description'.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE) 
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={'page_data':cleaned_text})
        try:
            json_parser = JsonOutputParser()
            json_result = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs. ")
        return json_result if isinstance(json_result, list) else [json_result]
    
    def write_email(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """ 
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Vishnu, a business development executive at VTG. VTG is an AI & Software Consulting company.
            Your job is to write a cold email to the client regarding the job mentioned above describing the role and fulfilling their needs.
            Also add the most relevat ones from the following links to showcase VTG's portfolio: {link_list}
            Remember you are Vishnu, BDE at VTG.
            ### EMAIL (NO PREAMBLE):
            
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return (res.content)

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))