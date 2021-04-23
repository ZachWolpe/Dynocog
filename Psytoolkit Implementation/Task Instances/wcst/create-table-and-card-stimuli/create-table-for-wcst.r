# the table of wcst.psy is quite long
# the table contains all trial information
# it was created with this script
# run it in R if you want to remake the table (which is not necessary)
# then copy and past the created table to the wcst.psy table
# ----------------------------------------------------------------------

shapes=c("circle","triangle","cross","star")
nums  =1:4
cols  =c("red","green","blue","yellow")

a=0
s=matrix(ncol=5,nrow=64)
for( i in 1:4 )
  for ( j in 1:4 )
    for ( k in 1:4 )
  {
    stimulus=paste(shapes[i],nums[j],cols[k],sep="")
    carddescription=paste( "\"" , ( paste( shapes[i],nums[j],cols[k] ) ) , "\"" ,sep="")
    z=c(stimulus, i,j,k,carddescription)
    a=a+1
    s[a,]=z
  }

tasks = c( "\"shape\"","\"number\"","\"color\"" )

# take out sample cards
s=s[ s[,1]!="circle1red"&s[,1]!="triangle2green"&s[,1]!="cross3blue"&s[,1]!="star4yellow",]

trials=sample( 1:60 )

s=s[trials,] # now the cards are mixed

xx=
rbind( s[ 1:10,c(1,2)],
       s[11:20,c(1,3)],
       s[21:30,c(1,4)],
       s[31:40,c(1,2)],
       s[41:50,c(1,4)],
       s[51:60,c(1,3)] )

##--------

blocklength = 10
blocks      = 6
ts = c(1,2,3,1,2,3) # tasks 1/2/3 as in variable "tasks"

x=matrix(nrow=blocks*blocklength,ncol=5)
#rownames(x)=rep( "",blocklength*blocks )
for ( i in 1:blocks )
  {
    rep = 1:blocklength
    name= rep( tasks[ ts[ i ] ] , blocklength )
#    a=cbind( s[ (i-1)*blocklength : i*blocklength , rep( ts+1 , blocklength )  )
    choosesequence = 1+((i-1)*blocklength ): (i*blocklength-1)
    x[choosesequence,]=cbind( s[choosesequence, c(1,ts[i]+1)] , rep ,name ,s[choosesequence,5])
print(x)
                                        #    print( cbind( a, rep, name ))
  }

cat("# table\n")
print.table(x)

print.data.frame(as.data.frame(x),row.names=F)

