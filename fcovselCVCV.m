%% Double cross-validation fCovSel
% Assumes that inner cross-validation is a subset of outer cross-validation
% to reuse segments.
%		1='Full crossvalidation (leave one out)'
%		2='Random crossvalidation (samles are randomly picked for each segment)'
%		3='Systematic crossvalidation 111..-222..-333.. etc.
%		4='Systematic crossvalidation 123..-123..-123.. etc.
%       5='Vector of segments'
function [RMSECVCV,YpredsCV, RMSECV,VarsCVCV,VarsCV, Ypreds, ncompi] = fcovselCVCV(X,y, A, modes, type,segments,stratify)
% ----------------------------------------------------------
% ------------------ PM fcovsel (used KHL codes) 2022 ------
% ----------------------------------------------------------
% ------------- Solution of the fCovSel-problem ------------
% ------------- Double cross-validation style --------------

% Initializations
if nargin < 7
    stratify = [];
end

dims = size(X);
if size(dims,2)>2
   X = reshape(X,[dims(1) prod(dims(2:end))]);
end

n = size(X,1);
[segsOuter, segsInner, segments] = segmentify(n, type, segments, stratify);
nseg  = max(segsOuter);
sumX = sum(X,1);
sumy = sum(y,1);
% Sum within segment (reused in inner loop)
sumSegX = (segsInner==0)*X;
sumSegy = (segsInner==0)*y;
segLength = zeros(nseg,1);
for i=1:nseg
    segLength(i) = sum(segsOuter==i);
end
% Means without i-th sample set
Xc = bsxfun(@times,bsxfun(@minus,sumX,sumSegX),1./(n-segLength));
yc = bsxfun(@times,bsxfun(@minus,sumy,sumSegy),1./(n-segLength));
yO = y;
for j = 1:size(y,2)
    Ypreds{j} = zeros(n,A);     % Storage of predictions
end
n2 = false(1,n);
T  = zeros(n,A); q = zeros(size(y,2),A);
ncompi = zeros(nseg,1);
for i=1:nseg
    % Inner cross-validation loop
    segsi = segsInner(i,:);
    nsegi = segments-1;
    ni    = length(segsi);
    Xi    = X;
    yi    = yO; yi(segsOuter==i,:) = 0;
    % Means without i-th sample set
    Xci = bsxfun(@times,bsxfun(@minus,sumX-sumSegX(i,:),sumSegX),1./(n-segLength)); 
    yci = bsxfun(@times,bsxfun(@minus,sumy-sumSegy(i,:),sumSegy),1./(n-segLength)); 
    yci(i,:) = [];
    yOi  = yi;
    for j = 1:size(y,2)
        Ypredsi{j} = zeros(ni,A);   % Storage of predictions
    end
    n2i = false(1,ni); n2i(1,segsOuter==i) = true; % Lookup without outer test
    T_i  = zeros(ni,A); qi = zeros(size(y,2),A);
    for k=1:nsegi
        inds2i = n2i;
        inds2i(segsi==k) = true;
        yi  = yOi-yci(k,:);
        yi(inds2i,:) = 0;
        X_tempi = Xi;
        X_tempi = X_tempi-Xci(k,:);
        X_temp_vi = X_tempi(segsi==k,:);
        X_tempi(inds2i,:) = 0;   
        for a = 1:A
            v = sum((X_tempi'*yi).^2 ,2);
            if size(dims,2)>2
                if modes<2
                    [~,ind]=max(v);
                    temp_VarsCV(k,a)=ind;
                    Wi(:,a) = zeros(size(v)); Wi(ind,a) = 1;
                    ti = X_tempi*Wi(:,a);
                else
                    v = reshape(v,dims(2:end));
                    [wi{1},~,wi{2}] = svds(v,1);
                    [~,ind]=max(abs(wi{modes-1}));
                    temp_VarsCV(k,a)=ind;
                    wi{modes-1} = zeros(size(wi{modes-1})); wi{modes-1}(ind) = 1;
                    ti = X_tempi*reshape(wi{1}*wi{2}',[prod(dims(2:end)) 1]);
                    Wi(:,a) = reshape(wi{1}*wi{2}',[prod(dims(2:end)) 1]);
                end
            else
                    [~,ind]=max(v);
                    temp_VarsCV(k,a)=ind;
                    Wi(:,a) = zeros(size(v)); Wi(ind,a) = 1;
                    ti = X_tempi*Wi(:,a);
            end
            if a > 1
                ti = ti - T_i(:,1:a-1)*(T_i(:,1:a-1)'*ti);
            end
            ti = ti/norm(ti); T_i(:,a) = ti;
            % ------------------- Deflate y ----------------------
            qi(:,a) = yi'*ti; 
            yi = yi - ti*(ti'*yi);
        end
        % ---------- Calculate predictions -------------
        for j = 1:size(q,1)
            betai{j}  = cumsum(bsxfun(@times,Wi/triu((X_tempi'*T_i)'*Wi), qi(j,:)),2);
            Ypredsi{j}(segsi==k,:) = X_temp_vi*betai{j};
        end
    end
    VarsCV(i,:)= mode(temp_VarsCV,1);
    yci = [zeros(size(q,1),1)';yci]; %#ok<AGROW>
    for j = 1:size(q,1)
        Ypredsi{j} = bsxfun(@plus, [zeros(ni,1) Ypredsi{j}], yci(segsi+1,j));
        RMSECVi{j} = sqrt(((segsOuter~=i)*(bsxfun(@minus,yOi(:,j),Ypredsi{j}).^2))./(n-segsOuter(i)));
    end

    [~,ncompi(i,1)] = min(mean(cell2mat(RMSECVi'),1));
    
    % Prediction using optimal number of components
    inds2 = n2;
    inds2(segsOuter==i) = true;
    % Compute X*X' with centred X matrices excluding observation i
    y  = yO-yc(i,:);
    y(inds2,:) = 0;
    X_temp = X;
    X_temp = X_temp-Xc(i,:);
    X_temp_v = X_temp(segsOuter==i,:);
    X_temp(inds2,:) = 0; 
    
    for a = 1:A
        v = sum((X_temp'*y).^2 ,2);
        if size(dims,2)>2
            if modes<2
                [~,ind]=max(v);
                VarsCVCV(i,a)=ind;
                W(:,a) = zeros(size(v)); W(ind,a) = 1;
                t = X_temp*W(:,a);
            else
                v = reshape(v,dims(2:end));
                [w{1},~,w{2}] = svds(v, 1);
                [~,ind]=max(abs(w{modes-1}));
                VarsCVCV(i,a)=ind;
                w{modes-1} = zeros(size(w{modes-1})); w{modes-1}(ind) = 1;
                t = X_temp*reshape(w{1}*w{2}',[prod(dims(2:end)) 1]);
                W(:,a) = reshape(w{1}*w{2}',[prod(dims(2:end)) 1]);
            end
        else
            [~,ind]=max(v);
            VarsCVCV(i,a)=ind;
            W(:,a) = zeros(size(v)); W(ind,a) = 1;
            t = X_temp*W(:,a);
        end
        
        if a > 1
            t = t - T(:,1:a-1)*(T(:,1:a-1)'*t);
        end
        t = t/norm(t); T(:,a) = t;
        % ------------------- Deflate y ----------------------
        q(:,a) = y'*t; y = y - t*(t'*y);
    end
    % ---------- Calculate predictions -------------
    for j = 1:size(q,1)
        beta{j}  = cumsum(bsxfun(@times,W/triu((X_temp'*T)'*W), q(j,:)),2);
        Ypreds{j}(segsOuter==i,:) = X_temp_v*beta{j};
    end
end    
for j = 1:size(q,1)
    Ypreds{j} = bsxfun(@plus, [zeros(n,1) Ypreds{j}], yc(segsOuter,j));
    RMSECV{j} = sqrt(mean(bsxfun(@minus,yO(:,j),Ypreds{j}).^2));
end
for j = 1:size(q,1)
    YpredsCV{j} = zeros(n,1);
    for i=1:nseg
        YpredsCV{j}(segsOuter==i,1) = Ypreds{j}(segsOuter==i, ncompi(i));
    end
RMSECVCV{j} = sqrt(mean(bsxfun(@minus,YpredsCV{j},yO(:,j)).^2));
VarsCVCV = mode(VarsCVCV,1);
VarsCV = mode(VarsCV,1);
end

%% Create segment combinations                                             ------ TODO: should inner segments be a subset of outer segments? -------
function [segsOuter, segsInner, segments] = segmentify(N, type, segments, stratify)
% N    : number of samples
% type - type of crossvalidation
%		1='Full crossvalidation (leave one out)'
%		2='Random crossvalidation (samles are randomly picked for each segment)'
%		3='Systematic crossvalidation 111..-222..-333.. etc.
%		4='Systematic crossvalidation 123..-123..-123.. etc.
%       5='Vector of segments'
% segments - number of segments
% stratify - vector of classes to stratify over
% 
% segsOuter - vector of indices
% segsInner - cell of vectors of indices assuming removal of chosen outer
if type == 1 % LOO
    segsOuter = 1:N;
    segments  = N;
elseif type == 2 % Random segments
    if ~isempty(stratify)
        stop('Not implemented yet')
    else
        segsOuter = repmat(1:segments,1,ceil(N/segments));
        r  = randperm(N);
        segsOuter = segsOuter(r);
    end
elseif type == 3 % Consecutive
    if ~isempty(stratify)
        stop('Not implemented yet')
    else
        segsOuter = repmat(1:segments,1,ceil(N/segments)); segsOuter = sort(segsOuter(1:N));
    end
elseif type == 4 % Interleaved
    if ~isempty(stratify)
        stop('Not implemented yet')
    else
        segsOuter = repmat(1:segments,1,ceil(N/segments)); segsOuter = segsOuter(1:N);
        segsOuter = segsOuter(1:N);
    end
elseif type == 5 % User chosen
    if ~isempty(stratify)
        stop('Not implemented yet')
    else
        if size(segments,2) == 1
            segsOuter = segments';
        else
            segsOuter = segments;
        end
        segments  = max(segsOuter);
    end
end
segsInner = zeros(segments,N);
for i=1:segments
    segs = segsOuter;
    segs(segs==i) = 0;
    segs(segs>i)  = segs(segs>i)-1;
    segsInner(i,:) = segs;
end
