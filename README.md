# CWF_burn_die_causal_rail_evaluation
This repository stores the ML code for the CWF burn die causal rails evaluation

Author : Sutirtha Chowdhury, STTD, IPE WWID : 12140215

=============================================================================


Background:

CWF Top die showing pull-pad issue due to die burn after electrical die testing. This die burnings are occuring due to ISVM test (measuring current with particular voltage). This die burns are very impactful towards our probecard, where the burn dies causing extensive probecard damage. The test program flow below (typical TPI : SIUP flowplan) showing multiple VCC rails involvement and PDE / test program team wants to which rails are relevant / sensitive towards a die to be burned. In other word, can we find the importance of rails that correlates towards more burn. Identifying sensitive rails would be easier for PDE/test program team to build up better corrleation between sesitive rails and limit evaluation to implement the ADTL kills. These way we can minimze our probe card to have damage and reduce the SIU return rate. 

- Test program Flow image :
![Screenshot of the application](C:\Users\sutirtha\Downloads\TPI_SIUP_flow.png)

Approach :



