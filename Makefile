#
# Testing helpers.
#

.PHONY : test
test :
	pytest --color=yes -rf --durations=40
