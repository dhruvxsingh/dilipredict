cat requirements.txt 
cat setup.
cat setup.py 
cd dilipredict/
ls
ls dilipredict/nets/
ls nets/
cat models.py 
cat pipelines.py 
cd ..
ls *.py
find . -name "main*.py" -o -name "train*.py" -o -name "example*.py" -o -name "demo*.py"
cat dilipredict/nets/__init__.py
head -100 dilipredict/nets/beitv2.py
# Check the image_loader.py
cat dilipredict/image_loader.py
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
cat > requirements_updated.txt << EOF
numpy
torch>=2.0.0
torchvision
einops
pillow
scikit-learn
EOF

pip install -r requirements_updated.txt
touch demo.py
python demo.py
git status
