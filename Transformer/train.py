from model import build_transformer

def main():
    model = build_transformer(32, 32, 512, 6, 8, 0.1, 2048)
    print(model)
    
main()