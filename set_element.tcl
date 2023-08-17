# Load a single pdb and add the element to each atom name
# this step is required to generate the receptor pdbqt with autodick vina!!!
​
# To run this script: vmd -dispdev text -e set_element.tcl -args input_pdb_file output_pdb_file
​
set input_pdb_file [lindex $argv 0]
set output_pdb_file [lindex $argv 1]
​
​
mol new $input_pdb_file
​
set all [atomselect top all]
​
set indexes [$all get index]
​
foreach indexx $indexes {
​
    set atom [atomselect top "index $indexx"]
    
    set elementx [lindex [split [$atom get name] {}] 0]
​
    if {[string compare [$atom get name] "F1"] == 0} {
​
        set elementx F
        puts $indexx
​
    } 
​
    if {[string compare [$atom get name] "FE"] == 0} {
​
        set elementx Fe
        puts $indexx
​
    } 
​
    $atom set element $elementx
    $atom delete
}
​
$all writepdb $output_pdb_file
mol delete all
​
exit
