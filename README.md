# CWF_burn_die_causal_rail_evaluation
This repository stores the ML code for the CWF burn die causal rails evaluation

Author : Sutirtha Chowdhury, STTD, IPE WWID : 12140215

=============================================================================


Background:

CWF Top die showing pull-pad issue due to die burn after electrical die testing. This die burnings are occuring due to ISVM test (measuring current with particular voltage). This die burns are very impactful towards our probecard, where the burn dies causing extensive probecard damage, causing > 10 probecard return (normalized per wafer). The target is to met ~5 probecard /wafers. The test program flow below (TPI : SIUP flowplan) showing multiple VCC rails involvement in the testing, and PDE / test program team wants to identify which rails are relevant / sensitive towards a die to be burned. In other word, can we find the importance of rails that correlates towards burn. Identifying sensitive rails would be easier for PDE/test program team to build up better corrleation model between VCC rails and which will essentially lead to optimized limit evaluation to implement the ADTL kills. These way we can minimze our probe card to have damage and reduce the SIU return rate.  

- Test program Flow image :
![image](https://github.com/user-attachments/assets/b3cc109a-6f1e-41cd-a9c0-ba16377fbb3c)


Approach :

Here we implement a Machine Learnig (ML) agorithm : Random Forest Classifier (Tree Ensemble) on those VCC rails to calculate the permutance importance score to evaluate the sesitivity towards a burnt die prediction. 

- Algorithm Detail :
  - Input feature :
    We have the Raw Current (I) data for a measured voltage.<br>
    **Total Input feature** : 48 VCC rails. <br>
    **Total Example (no. of units / dies)** : 50K units.
  - Output Label :
    Burnt / not-burnt units <br>
    **burnt labeled as 0** and **not-burnt labeled as 1**
  - Algorithm architecture :
    ![image](https://github.com/user-attachments/assets/299b5bec-6d66-49f2-a5f6-33f3548e43ed)


