This is a learning project I'm working on. 
I'm using ChatGPT as a teacher to learn the following topics:

- Creating a simple neural network that takes a Boolean expression as input and outputs a truth table. 
- Extending this to image recognition so that the program can receive an image of an expression and generate a value table from it
- Offloading math operations to the GPU (AMD)
- Outputting the value table as an image (?)

All of this without being shown source code or pseudocode, since I want to do the programming part myself and truly understand how it all works. And you don’t learn anything from copy-pasting.

I’m specifically using a problem for the AI that could actually be solved with a standard algorithm, since this allows me to easily create my own training data. And since the goal of this program isn’t real-world application but is solely for learning purposes, that’s not a problem.

Before, I was using Gemini (with a Plus subscription) and its training tool to study. But I quickly hit the limits of Gemini there, since it kept forgetting or mixing things up. I know this is a common problem—after all, memory isn’t unlimited—and that you can “remind” the AI of things with the right prompts, but I still wanted to try an alternative. So now I’m using ChatGPT, and I’m satisfied so far. It explains the math formulas a bit better, and so far it hasn’t forgotten anything (though I know that will happen eventually). 

Translated with DeepL.com (free version)

Transferring the chat went a little differently than I expected. Instead of simply copying a link to my Gemini chat and pasting it into ChatGPT, I had to copy the entire (!) chat and paste it into ChatGPT, since it seems you can’t view the chat without a Google account. From there, though, ChatGPT was able to take over immediately.
I recommend saving the old chat to a text file and giving that to ChatGPT instead of sending the chat directly as a message. ChatGPT can read the file much faster.

The project is far from finished. 

What has been done so far:

Current status:

- Implemented a basic data encoding pipeline for Boolean expressions.
- Added token/ID handling for variables, operators, parentheses, and padding.
- Prepared one-hot encoding / flattening for neural network input.
- Implemented flattened weight storage for better performance and future GPU-readiness.
- Added Xavier/Glorot weight initialization.
- Implemented forward propagation for layers.
- Added loss calculation using Mean Squared Error.
- Added MSE derivative calculation.
- Added sigmoid activation and sigmoid derivative.
- Added delta calculation for output neurons.
- Added weight and bias gradient calculation.
- Created an Optimizer class.
- Implemented basic Gradient Descent with a configurable learning rate.
- First successful end-to-end training test completed:
        The network learned to map the encoded expression "A AND B" to its truth-table output [0, 0, 0, 1].
- Added Trainer class.
- Added activation saving during forward pass.
- Implemented hidden-layer delta calculation.
- Implemented multi-layer backpropagation.
- Added training support for hidden layers.
- Added multiple Boolean expression training samples.
- Added console output for predictions, targets and MSE.
- Added simple Timer class.
- Successfully trained multiple encoded Boolean formulas to truth-table outputs.

Next step:
Automate dataset generation with a Boolean expression evaluator and formula generator in a separate repository. Link will be added later.
