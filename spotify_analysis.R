# analysis of music harmony

# extract csv of your playlists with https://www.chosic.com/spotify-playlist-sorter/


# dependencies ----
library(readr) # na read.csv
library(ggplot2)
library(dplyr) # kvuli funkcim
library(tidyr) 


# Open data ----

setwd("~/Library/CloudStorage/OneDrive-MUNI/Grafy Mapy/Hudba harmonie")

#nase spolecne 4/2023
data <- read_csv("23-04-16 - Spotify playlists/Lenka + jarogumulec.csv")
# liked cca 2000 do dobna 23
data <- read_csv("23-04-16 - Spotify playlists/Liked Songs.csv")




# major and minor DEPREC ---- 
# extract major/minor from Key column
data$MajorMinor <- ifelse(grepl("Minor", data$Key), "minor", "major")
# count the number of major/minor keys in the data table
major_minor_count <- table(data$MajorMinor)

# create a data frame with the counts and key type (major/minor)
major_minor_df <- data.frame(
  MajorMinor = names(major_minor_count),
  Count = as.numeric(major_minor_count),
  stringsAsFactors = FALSE
)

# create the bar chart
ggplot(major_minor_df, aes(x=MajorMinor, y=Count, fill=MajorMinor)) + 
  geom_col() + 
  labs(x="Key Type", y="Count", title="Major/Minor Key Distribution")


# camelot notation freq -----


#simple
ggplot(data, aes(x = Camelot)) +
  geom_bar(fill = "lightblue") +
  labs(x = "Camelot", y = "Count", title = "Camelot distribution") +
  theme_bw()


# complex - show key together ----

# calculate counts
counts <- data %>% count(Camelot)
#define all keys
keys <- c("1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B", "6A", "6B", "7A", "7B", "8A", "8B", "9A", "9B", "10A", "10B", "11A", "11B", "12A", "12B")
# fill the missing keys with 0 counts
counts <- counts %>% complete(Camelot = keys, fill = list(n = 0))
counts$camelotkey <- as.integer(substr(counts$Camelot, 1, nchar(counts$Camelot)-1))
counts$minormajor <- substr(counts$Camelot, nchar(counts$Camelot), nchar(counts$Camelot))


ggplot(counts, aes(x = camelotkey, y = n, fill = minormajor)) +
  geom_col(position = "stack", width = 0.9) +
  scale_fill_manual(values = c("lightblue", "pink")) +
  labs(x = "", y = "", title = "Camelot distribution", fill = "") +
  scale_x_continuous(breaks = 1:12) +
  theme_bw() +
  theme(legend.position = "right") +
  coord_polar(start = 0.25)
