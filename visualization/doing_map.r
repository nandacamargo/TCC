geo <- read.csv("geo.dat", header=TRUE)
head(geo)

library(ggmap)
library(plyr)
map <- get_map(location = 'United States of America', zoom = 4)


info <- ddply(geo, c("state"), summarise, quant = length(state), lat = mean(lat), lon = mean(lon))
head(info)

mapPoints <- ggmap(map) + geom_point(aes(x = lon, y = lat, size = quant), data = info, alpha = .5)
mapPoints

mapPoints <- ggmap(map) + geom_point(aes(x = lon, y = lat, size =(quant), color=state), data = info, alpha = 1)
mapPointsLegend <- mapPoints + scale_size_area(name = "Cidades com mais tweets")
mapPointsLegend