# cs1470-final
UnRembrandt -- Neural Painting to Photo decoder

Plan for Preprocessing
-Use all 7.2k faces
-Use 50 style references (10-40 breakdown between generic and portrait)
-Loop through all 7.2k faces. For each face, pick a random style reference from the (e.g.) 50 references. If the reference chosen is generic, it will be used with the generic model to generate an output image. If the reference chosen is a portrait, it will be used with the portrait model to generate an output image. If this is successful and time allows, we will loop through the 7.2k faces another time, generating a second image.

Sebastian's TODO: Get the portrait model working. Script the autoencoder architecture
Daniel's TODO: Search for more style reference images (for use with the generic model)
Harys's TODO: Also try (in tandem with Sebastian) to get the portrait model working
