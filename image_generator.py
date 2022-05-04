from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import MolFromSmiles
from PIL import ImageOps, Image
from indigo.renderer import IndigoRenderer
from indigo import Indigo
import numpy as np
import random
import io

# Liberaries :
# 0 - Indigo
# 1 - RDKit
# 2 - OpenEye -> need license for this :-(

class ImageGenerator():
    def __init__(self,grayScale=False):
        self.library_used = None
        self.imageSize = (224,224)
        self.bondlength = 100
        self.thickness = 1.0
        self.bondlinewidth = 1.0
        self.renderatomids = False
        self.colorrendering = True
        self.smile = ""
        self.molecule = None
        self.image = None
        self.grayScale = grayScale
        self.indigo = Indigo()
        self.renderer = IndigoRenderer(self.indigo)

    def RandSize(self):
        if self.library_used == "rdkit":
            size = np.random.randint(190,2500)
        else:
            size = np.random.randint(190,500)
        self.imageSize = (size, size)

    def UpdateBondLength(self):
        self.bondlength = np.random.randint(50,150)

    def UpdateThickness(self):
        self.thickness = np.random.uniform(0.85,1.15)
    
    def BondLineWidth(self):
        self.bondlinewidth = np.random.uniform(0.6,1.6)
    
    def RenderAtomIds(self):
        self.renderatomids = (np.random.uniform()>0.8)
    
    def ColorRendering(self):
        self.colorrendering = (np.random.uniform()>0.2)

    def UpdateSmile(self,smile):
        self.smile = smile

    def Convert2Gray(self):
        self.image = ImageOps.grayscale(self.image)

    def UseOpenEye(self):
        return True

    def UseRDKit(self):
        ''' Save PIL Image of the Molecule
        '''
        self.molecule = MolFromSmiles(self.smile)
        self.image = MolToImage(self.molecule,size=self.imageSize)

    def UpdateIndigoArguments(self):
        self.UpdateBondLength()
        self.UpdateThickness()
        self.ColorRendering()
        self.BondLineWidth()
        self.RenderAtomIds()

    def SetIndigoOptions(self):
        self.UpdateIndigoArguments()
        self.indigo.setOption("render-output-format","png")
        self.indigo.setOption("render-background-color","255,255,255")
        self.indigo.setOption("render-image-size",self.imageSize[1],self.imageSize[0])
        self.indigo.setOption("render-bond-length",str(self.bondlength))
        self.indigo.setOption("render-relative-thickness",self.thickness)
        self.indigo.setOption("render-coloring",self.colorrendering)
        self.indigo.setOption("render-bond-line-width",self.bondlinewidth)
        self.indigo.setOption("render-atom-ids-visible",self.renderatomids)

    def UseIndigo(self):
        self.molecule = self.indigo.loadMolecule(self.smile)
        self.SetIndigoOptions()
        bimage = self.renderer.renderToBuffer(self.molecule)
        self.image = Image.open(io.BytesIO(bimage))

    def UpdateSelf(self,smile):
        self.UpdateSmile(smile)
        self.RandSize()

    def genImage(self,smile):
        self.library_used = np.random.choice(["rdkit", "indigo"], p=[0.25, 0.75])
        if len(smile) > 60:
            self.library_used = "rdkit"
        
        try:
            self.UpdateSelf(smile)
            if self.library_used == "rdkit":
                self.UseRDKit()
            else:
                self.UseIndigo()

        except Exception as e:
            # print(e)
            self.UpdateSelf(smile)
            self.UseRDKit()
            
        if(self.grayScale): 
            self.Convert2Gray()
        return self.image


if __name__=='__main__':
    ig = ImageGenerator()
    img = ig.genImage("O=C(COC(=O)COc1ccc(Cl)cc1)Nc1ccc(Cl)cn1")
    # img.save("./image1.png")
    # img = ig.genImage("CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C")
    # img.save("./image2.png")