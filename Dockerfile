FROM ubuntu:18.04 AS base
LABEL maintainer="support@charm-crypto.com"

RUN apt update && apt install --yes build-essential flex bison wget subversion m4 python3 python3-dev python3-setuptools python3-numpy libgmp-dev libssl-dev
RUN wget https://crypto.stanford.edu/pbc/files/pbc-0.5.14.tar.gz && tar xvf pbc-0.5.14.tar.gz && cd /pbc-0.5.14 && ./configure LDFLAGS="-lgmp" && make && make install && ldconfig
COPY ./ext /charm
RUN cd /charm && ./configure.sh && make && make install && ldconfig

FROM base AS testing 
# Helper class 
COPY ./qfehelpers.py .
# Test class
COPY ./test_qfehelpers.py .
# Run the tests
RUN python3 test_qfehelpers.py

FROM testing AS final 
COPY ./uqfer.py .
COPY ./benchmark.py .
CMD ["python3", "benchmark.py"]