# Run from the app/ folder.
up:        ## start (builds the image the first time)
	docker compose up -d --build

rebuild:   ## after ANY change to the Cadenza code, models or the app: rebuild + restart
	docker compose build
	docker compose up -d

rebuild-clean:  ## same, but ignore the build cache (re-download Clarity and the models)
	docker compose build --no-cache
	docker compose up -d

logs:
	docker compose logs -f cadenza

down:
	docker compose down

test:      ## run the unit tests inside a throwaway container
	docker compose run --rm --no-deps -v "$(CURDIR)/tests:/srv/tests" --entrypoint sh cadenza -c \
	  "pip install -q pytest httpx && python -m pytest -q /srv/tests"
