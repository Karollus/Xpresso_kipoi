#!/usr/bin/perl

open IN, "<yeastorfs.withIDs.txt";
while(<IN>){ chomp;
  @a=split /\t/;
  $id2orf{$a[0]} = $a[1];
}

$a=<>;
print $a;
while(<>){
  @a=split /\t/;
  $line=`grep -P "\$a[0]" yeastorfs.withIDs.txt`;
  chomp $line;
  print join("\t", $a[0], $id2orf{$a[0]}, @a[2..$#a]) if $id2orf{$a[0]} ne "";
}
