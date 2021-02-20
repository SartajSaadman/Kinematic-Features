# Kinematic-Features
Final project for master's program

## Background
Recently I have been working with a team of researchers (Lead by Prof. Michael Caliguri) exploring the relationship between kinematics of handwritten text and the ability of forensic document examiners to determine if two documents were written by the same writer.

A side debate has started among the team concerning whether or not the the statisticians and machine learning experts could build a reasonably accurate classifier that predicts three things at once 
1. The writer of a short note (40 writers)
2. Which of 6 phrase was written
3. What type/style of writing was used (Print or Script)
Based on the kinematic features recorded by MovAlyzeR.
  
## Data structure

The column Labels of the data are as follows (This is the basic output order from MovAlyzeR.) Please see the provide powerpoints and papers for details.

1. Group- The type of handwriting- either Cursive or print.
2. Subject is the writer of the handwriting sample.
3. Condition- Is the Phrase that is being written.
4. Trial is one handwriting sample
5. Segment is one part of the writing sample a 'stroke'.
6. Direction- if the stroke is up or down.
7. StartTime- When the segment was started to be written.
8. Duration- How long it has taken to complete the segment.
9. VerticalSize                      
10. PeakVerticalVelocity              
11. PeakVerticalAcceleration                     
12. HorizontalSize                   
13. StraightnessError                 
14. Slant                             
15. LoopSurface                       
16. RelativeInitialSlant              
17. RelativeTimeToPeakVerticalVelocity
18. RelativePenDownDuration           
19. RelativeDurationofPrimary         
20. RelativeSizeofPrimary             
21. AbsoluteSize                      
22. AverageAbsoluteVelocity           
23. Roadlength                       
24. AbsoluteyJerk                     
25. NormalizedyJerk                   
26. AverageNormalizedyJerkPerTrial    
27. AbsoluteJerk                     
28. NormalizedJerk                    
29. AverageNormalizedJerkPerTrial                                
30. NumberOfPeakAccelerationPoints   
31. AveragePenPressure                


## Data Sets

You will be provided with 

1. labeled.csv
2. example of unlabeled data to design your predictions on my unlabeled data set on my computer. (There are two documents to classify.)

## Final work
1. You will need to provide an estimate of the accuracy of the classifier you build when applied to the third dataset of that I will use your code on. (Please make sure that you explain how to get your predictions.)
2. Your knowledge of the material in Stat 601, Stat 602, and the methods used in constructing your classifiers as demonstrated by the white paper component of the exam.
