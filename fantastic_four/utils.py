def decode_prediction(preds):
    # Convert model output to text
    # Replace with your own decoding logic (CTC decode, char mapping, etc.)
    preds = preds.squeeze().cpu().numpy()
    text = "".join([chr(int(p % 94) + 32) for p in preds])
    return text
