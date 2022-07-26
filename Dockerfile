FROM takuseno/d3rlpy:latest
COPY ./requirements.txt /workspace/additional_requirements.txt
RUN pip install -r additional_requirements.txt
CMD ["/opt/conda/bin/python", "/workspace/run.py"]