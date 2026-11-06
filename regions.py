"""Exact 8-connected predicted-oil regions, in zero-based image coordinates."""
import csv
import numpy as np

FIELDS=['region_id','pixel_count','row_min','col_min','row_max','col_max','centroid_row','centroid_col']

def extract(classes,path):
    # Run-length union/find avoids a Python object per oil pixel.
    parent=[];stats=[]
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):
        a,b=find(a),find(b)
        if a==b:return a
        if stats[a][0]<stats[b][0]:a,b=b,a
        parent[b]=a
        x,y=stats[a],stats[b]
        stats[a]=[x[0]+y[0],min(x[1],y[1]),min(x[2],y[2]),max(x[3],y[3]),max(x[4],y[4]),x[5]+y[5],x[6]+y[6]]
        return a
    previous=[]
    for row in range(classes.shape[0]):
        line=(np.asarray(classes[row])==1).astype(np.int8)
        changes=np.flatnonzero(np.diff(np.concatenate(([0],line,[0]))))
        current=[];first=0
        for start,end_exclusive in zip(changes[::2],changes[1::2]):
            start,end=int(start),int(end_exclusive)-1
            count=end-start+1
            identifier=len(parent);parent.append(identifier)
            stats.append([count,row,start,row,end,row*count,(start+end)*count//2])
            while first<len(previous) and previous[first][1]<start-1:first+=1
            pos=first
            while pos<len(previous) and previous[pos][0]<=end+1:
                identifier=join(identifier,previous[pos][2]);pos+=1
            current.append((start,end,identifier))
        previous=current
    components=[]
    for i,item in enumerate(stats):
        if find(i)==i:
            area,r0,c0,r1,c1,sumr,sumc=item
            components.append({'pixel_count':area,'row_min':r0,'col_min':c0,'row_max':r1,'col_max':c1,
                'centroid_row':sumr/area,'centroid_col':sumc/area})
    components.sort(key=lambda item:(-item['pixel_count'],item['row_min'],item['col_min']))
    with open(str(path),'w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()
        for i,item in enumerate(components,1):writer.writerow(dict(item,region_id=i))
    return {'oil_region_count':len(components),'largest_oil_regions':[dict(item,region_id=i) for i,item in enumerate(components[:20],1)],
        'coordinate_convention':'zero-based row,col; top-left origin; bounding-box maxima inclusive; x=col,y=row',
        'region_connectivity':8,'region_note':'Regions are model predictions, not verified physical spill boundaries or geographic coordinates'}
