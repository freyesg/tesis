upgrade:
	git pull origin master

git:
	git add .
	git commit --author="freyesg <felipe.reyesg@usach.cl>" -m "ACTUALIZACIÓN $(shell date +%FT%T%Z)"
	git push origin master
