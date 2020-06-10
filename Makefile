DOCKER_IMAGE := egliu/faiss-docker
FAISS_VERSION := 1.6.3

image:
	docker build \
		--build-arg IMAGE=ubuntu:16.04 \
		--build-arg FAISS_CPU_OR_GPU=cpu \
		--build-arg FAISS_VERSION=$(FAISS_VERSION) \
		--tag $(DOCKER_IMAGE):$(FAISS_VERSION)-cpu \
		--tag $(DOCKER_IMAGE) .

	docker build \
		--build-arg IMAGE=nvidia/cuda:10.0-runtime-ubuntu16.04 \
		--build-arg FAISS_CPU_OR_GPU=gpu \
		--build-arg FAISS_VERSION=$(FAISS_VERSION) \
		--tag $(DOCKER_IMAGE):$(FAISS_VERSION)-gpu .

release:
	docker push $(DOCKER_IMAGE)
	docker push $(DOCKER_IMAGE):$(FAISS_VERSION)-cpu
	docker push $(DOCKER_IMAGE):$(FAISS_VERSION)-gpu

baseImage-cpu=$(DOCKER_IMAGE):$(FAISS_VERSION)-cpu
baseImage-gpu=$(DOCKER_IMAGE):$(FAISS_VERSION)-gpu

dev-cpu:
	@baseImage=${baseImage-cpu} bash scripts/start-dev-cpu.sh

dev-gpu:
	@baseImage=${baseImage-gpu} bash scripts/start-dev-gpu.sh

test-image-gpu:
	docker build \
		--build-arg IMAGE=egliu/faiss-docker:1.6.3-gpu \
		--tag $(DOCKER_IMAGE):$(FAISS_VERSION)-gpu-test \
		-f ./test-dockerfile/gpu/Dockerfile .