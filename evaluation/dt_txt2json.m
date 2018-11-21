clear
addpath('./cocoapi/MatlabAPI')
mainpath = '../output/valresults/resnet50';
subdir = dir(mainpath);
for j = 3 : length(subdir)
    ndt=0;
    dt_coco = struct();
    dt_path = fullfile(mainpath, subdir(j).name);
    res = load([dt_path,'/val_det.txt'],'%f');
    num_imgs = max(res(:,1));
    out = [dt_path,'/val_dt.json'];
    if exist(out,'file')
        continue
    end
    
    for i = 1:num_imgs
        bbs = res(res(:,1)==i,:);
        for ibb=1:size(bbs,1)
            ndt=ndt+1;
            bb=bbs(ibb,2:6);
            dt_coco(ndt).image_id=i;
            dt_coco(ndt).category_id=1;
            dt_coco(ndt).bbox=bb(1:4);
            dt_coco(ndt).score=bb(5);
        end
    end
    dt_string = gason(dt_coco);
    fp = fopen(out,'w');
    fprintf(fp,'%s',dt_string);
    fclose(fp);
end