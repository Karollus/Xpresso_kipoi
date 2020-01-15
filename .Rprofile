#library(utils)
#library(stats)
#library(datasets)
#library(grDevices)
#library(graphics)
#library(methods)

args = commandArgs(trailingOnly = T)
options(repos = "http://cran.us.r-project.org")

head <- function(x, y = 5) { base::print(utils::head(x, y)) }
say <- function(...) { base::print(paste(...)) }
#exit <- function (save="no", ...) { quit(save=save, ...) }

tryCatch({options(width = as.integer(Sys.getenv("COLUMNS")))}, error = function(err) {options(width=236)})

.Last <- function(){
    if (!any(commandArgs() == '--no-readline') && interactive()) {
    	require(utils)
    	try(savehistory(".Rhistory"))
    }
}

error.bar <- function(x, y, upper, lower=upper, length=0.1,...){
	if(length(x) != length(y) | length(y) !=length(lower) | length(lower) != length(upper))
	stop("vectors must be same length")
	arrows(x,y+upper, x, y-lower, angle=90, code=3, length=length, ...)
}

Zmedian=function(x){
	(x-median(x))/mad(x)
}

imageplots=function(x, ...){ fields::image.plot(cor(t(x),method='spearman', ...)) }
heatmaps=function(x) { gplots::heatmap.2(as.matrix(x),density.info="none", trace="none", breaks=seq(-2,2,0.1), symkey=FALSE, cexRow=0.8, scale="row", dendrogram="col", key=TRUE, Rowv=F, col=matlab::jet.colors(40), distfun=function(x){ as.dist(1-abs(cor(t(x),method='spearman'))) } ) }

#X11.options(type='dbcairo')

writefile = function(obj, x, ...){
	write.table(obj, file=x, quote=F, row.names=F, sep='\t', ...)
}

fastread = function(file, ...){
	data.table::fread(file,data.table=F,sep="\t", ...)
}

rbindall = function(dir, string='*', header=F, ...){
	names = list.files(dir,string)
	do.call(rbind, lapply( names, function(sample) {
		read.delim(sample, header=header)
	}))
}

mergeall = function(dir, string='*', by=1, all=T, header=F, ...){
	Reduce(function(x, y) merge(x, y, by=by, all=all, ...), lapply( names, function(sample) {
		read.delim(sample, header=header)
	}))
}