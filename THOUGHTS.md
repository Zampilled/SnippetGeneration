# My Thoughts 

My thoughts of the Snippet generation task are varied. I think my approach had several pros and several cons:

## Pros

### Randomized snippets
The snippets taken out of my code blocks where randomized by taken a position in the second quartile and a positon in the third quartile 
and removing all code inbetween. This prevents any bias in my selection while leaving enough context for the model.

### Full Code
The selected code were complete functions or classes from real repos. This allowed the code to not be trivial. This also meant that 
the code wasent made with the specific purpose of being understood by the model and improve its performance.

### Graphed Metrics
Instead of a flat mean metric the distribution of metrics is graphed and neatly displayed creating a deeper understanding of the performance.


## Cons

### Poor performance
In my dataset I did not get any exact matches and generally pretty low scores. This may be because I am taking too big of chunks or because my chunks come at random parts of code and not at the end of statements.
As this is my first code generation task I am not sure what the answer is but am interested in exploring it further

### Bias in selection 

There is bias in my selection of code snippets to use as I am aware of the task and would unconsciously choose based on what I would think the "best" code would be.

### Dispersed Runs
My runs of the model are not batched and are individual possibly creating a bottleneck in performance.

All around pretty fun to make and would be excited to do more of the same work ))