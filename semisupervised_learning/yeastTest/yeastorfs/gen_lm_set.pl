#!/usr/bin/perl

foreach $i (qw/A T C G/){
  foreach $j (qw/A T C G/){
    foreach $k (qw/A T C G/){
      push(@codons, "$i$j$k");
    }
  }
}

$a=<>;
@a=split /\t/, $a;
print join("\t", $a[0], "totCodons", @codons, @a[2..$#a]);
while(<>){
  @a=split /\t/;
  %counts = ();
  @orfseq = split / /, $a[1];
  foreach(@orfseq){
    $counts{$_}+=1;
  }
  chomp $line;
  print $a[0]."\t".($#orfseq+1)."\t";
  foreach(@codons){
    print int($counts{$_})."\t";
  }
  print join("\t", @a[2..$#a]);
}
