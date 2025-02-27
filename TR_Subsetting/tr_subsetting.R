library(data.table)

# Import the file that contains affiliation IDs of Turkey
aff.ids <- as.data.frame(fread("tr_affiliation_ids.txt", header=F))

# This function searches through MAG World dataset in the directory where 
# it is kept ("global directory") and extracts the files of the Turkey 
# subset and saves them in another directory ("local directory")

subset_tr <- function(affiliation.ids, local.directory, global.directory) {
  ids <- affiliation.ids
  # Subset Affiliations.txt
  setwd(global.directory)
  df <- as.data.frame(fread("Affiliations.txt", header=F))
  subdf <- df[df[,1] %in% ids[,1],]
  setwd(local.directory)
  write.table(subdf, file="Affiliations_TR.txt",
              quote=F, row.names=F, col.names=F, sep="\t")
  
  # Subset PaperAuthorAffiliations.txt
  setwd(global.directory)
  df <- as.data.frame(fread("PaperAuthorAffiliations.txt", header=F))
  subdf <- df[df[,3] %in% ids[,1],]
  pid <- unique(subdf[,1]); aid <- unique(subdf[,2])
  setwd(local.directory)
  write.table(subdf, file="PaperAuthorAffiliations_TR.txt",
              quote=F, row.names=F, col.names=F, sep="\t")
  
  # Subset Papers.txt
  setwd(global.directory)
  df <- as.data.frame(fread("Papers.txt", header=F))
  subdf <- df[df[,1] %in% pid,]
  setwd(local.directory)
  write.table(subdf, file="Papers_TR.txt",
              quote=F, row.names=F, col.names=F, sep="\t")
  
  # Subset PaperKeywords.txt
  setwd(global.directory)
  df <- as.data.frame(fread("PaperKeywords.txt", header=F))
  subdf <- df[df[,1] %in% pid,]
  fid <- unique(subdf[,3])
  setwd(local.directory)
  write.table(subdf, file="PaperKeywords_TR.txt",
              quote=F, row.names=F, col.names=F, sep="\t")
  
  # Subset PaperReferences.txt
  setwd(global.directory)
  df <- as.data.frame(fread("PaperReferences.txt", header=F))
  subdf <- df[df[,1] %in% pid,]
  setwd(local.directory)
  write.table(subdf, file="PaperReferences_TR.txt",
              quote=F, row.names=F, col.names=F, sep="\t")
  
  # Subset Authors.txt
  setwd(global.directory)
  df <- as.data.frame(fread("Authors.txt", header=F))
  subdf <- df[df[,1] %in% aid,]
  setwd(local.directory)
  write.table(subdf, file="Authors_TR.txt",
              quote=F, row.names=F, col.names=F, sep="\t")
  
  # Subset FieldsOfStudy.txt
  setwd(global.directory)
  df <- as.data.frame(fread("FieldsOfStudy.txt", header=F))
  subdf <- df[df[,1] %in% fid,]
  setwd(local.directory)
  write.table(subdf, file="FieldsOfStudy_TR.txt",
              quote=F, row.names=F, col.names=F, sep="\t")
  
  # Subset FieldOfStudyHierarchy.txt
  setwd(global.directory)
  df <- as.data.frame(fread("FieldOfStudyHierarchy.txt", header=F))
  subdf <- df[df[,1] %in% fid,]
  setwd(local.directory)
  write.table(subdf, file="FieldOfStudyHierarchy_TR.txt",
              quote=F, row.names=F, col.names=F, sep="\t")
}

# Example run
subset_tr(aff.ids, "/your/local/directory", "/your/global/directory")
