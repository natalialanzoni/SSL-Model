# k-NN Evaluation Uses CLS Token

## Confirmation: Yes, k-NN evaluation uses the CLS token

The k-NN evaluation in this DINO implementation correctly uses **only the CLS token** as specified in the DINO paper.

### Code Flow:

1. **In `knn_evaluate()` function** (line ~392):
   ```python
   feats = model(images, return_embedding=True)
   ```
   This calls the model with `return_embedding=True`

2. **In `VisionTransformer.forward()`** (lines 352-378):
   ```python
   def forward(self, x, return_embedding: bool = False):
       # ... process patches and add CLS token ...
       cls_tokens = self.cls_token.expand(batch_size, -1, -1)
       x = torch.cat((cls_tokens, x), dim=1)  # CLS token prepended as first token
       
       # ... pass through transformer blocks ...
       x = self.norm(x)
       cls_token_output = x[:, 0]  # Extract CLS token (first token)
       
       if return_embedding:
           return cls_token_output  # Return CLS token embedding only
   ```

3. **Key line**: `cls_token_output = x[:, 0]`
   - This extracts the **first token** which is the CLS token
   - When `return_embedding=True`, only this CLS token is returned
   - Patch tokens (x[:, 1:]) are NOT used for k-NN evaluation

### DINO Paper Specification:
> "For knn, we use only the [cls] token and retrieve the nearest neighbors for classification."

This implementation correctly follows this specification.
