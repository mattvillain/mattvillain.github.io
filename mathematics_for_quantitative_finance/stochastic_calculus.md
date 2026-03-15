# 3. Stochastic Calculus

Stochastic calculus is what happens when you want to define and integrate processes that happen in continuous time: you first define a process in continuous time, and that goes fairly smoothly (as we will do in a minute), but then you try using notions of integration from calculus, things go poorly. 

## Defining Stochastic Processes
Stochastic processes are random processes in continuous time. The word *processes* generally hints to time playing a role (so does dynamic!) and out of our models of time, we want $T \in \mathbb{R}$ to be choice. 

There are a few ways to start talking about stochastic calculus. Many introductions start with the notion of Wiener process or Brownian motion. Brown noticed that particles were vibrating under the microscope, even though they were inorganic (not living) and generally still. The *phenomenon* was called Brownian motion. Then we can *model* the phenomenon with a mathematical object called a *Wiener process*. Other texts, like *Grimmet and Stirzaker* start with the notion of a diffusion process. Others start with the quadratic variation. 

What matters is the following: so long as we are in discrete time, all the things that we usually care about in probability - like expectation, conditioning or finding a specific probability - may involve integrating a *rough path*, that is a path that is nowhere differentiable but everywhere continuous (roughly speaking). We will get into definitions. 

## Defining Stochastic Integration
So at the heart, stochastic calculus is about integration. 
