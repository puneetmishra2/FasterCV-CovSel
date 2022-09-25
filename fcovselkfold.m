function [RMSECV, Ypreds, vars] = fcovselkfold(X,y,A,cv,modes)
% --------------------------------------------------------------
% ---------------- PM fcovsel (Used KHL codes) 2022 ------------
% --------------------------------------------------------------
%- --------- Solution of the fcovsel kfold CV-problem ----------
nseg  = max(cv);
if size(cv,2) == 1
    cv = cv';
end
dims = size(X);
if size(dims,2)>2
   X = reshape(X,[dims(1) prod(dims(2:end))]);
end
n = size(X,1);
vars = zeros(nseg,A);
sumX = sum(X,1);sumy = sum(y,1);
segLength = zeros(nseg,1);
cvMat = zeros(nseg,n);
for i=1:nseg
    segLength(i) = sum(cv==i);
    cvMat(i,cv==i) = 1;
end
% Means without i-th sample set
Xc = bsxfun(@times,bsxfun(@minus,sumX,cvMat*X), ...
    1./(n-segLength));
yc = bsxfun(@times,bsxfun(@minus,sumy,cvMat*y), ...
    1./(n-segLength));
yO = y;                % Original response
Ypreds = cell(1,size(y,2));   % Storage of predictions
RMSECV = cell(1,size(y,2));
n2 = false(1,n);
T  = zeros(n,A); q = zeros(size(y,2),A);
for i=1:nseg
    inds2 = n2;
    inds2(cv==i) = true;
    y = yO-yc(i,:);
    y(inds2,:) = 0;
    X_temp = X;
    X_temp = X_temp-Xc(i,:);
    X_temp_v = X_temp(cv==i,:);
    X_temp(inds2,:) = 0;
    for a = 1:A
        v = sum((X_temp'*y).^2 ,2);
        if size(dims,2)>2
            if modes<2
                [~,ind] = max(v);
                vars(i,a) = ind;
                W(:,a) = zeros(size(v)); W(ind,a) = 1;
                t = X_temp*W(:,a);
            else
                v = reshape(v,dims(2:end));
                [w{1},~,w{2}] = svds(v, 1);
                [~,ind]=max(abs(w{modes-1}));
                vars(i,a) = ind;
                w{modes-1} = zeros(size(w{modes-1})); w{modes-1}(ind) = 1;
                t = X_temp*reshape(w{1}*w{2}',[prod(dims(2:end)) 1]);
                W(:,a) = reshape(w{1}*w{2}',[prod(dims(2:end)) 1]);
            end
        else
            [~,ind]=max(v);
            vars(i,a) = ind;
            W(:,a) = zeros(size(v)); W(ind,a) = 1;
            t = X_temp*W(:,a);
        end
        if a > 1
            t = t - T(:,1:a-1)*(T(:,1:a-1)'*t);
        end
        t = t/norm(t); T(:,a) = t;
        % ---------------- Deflate y ------------------
        q(:,a) = y'*t; y = y - t*(t'*y);
    end
    % ---------- Calculate predictions -------------
    for j = 1:size(q,1)
            beta  = cumsum(bsxfun(@times,W/triu((X_temp'*T)'*W), q(j,:)),2);
            Ypreds{j}(cv==i,:) = X_temp_v*beta;
    end
end
vars = mode(vars,1);
% Predicted 0-th component = mean(y) (per segment)
figure,
    for j = 1:size(q,1)
        Ypreds{j} = bsxfun(@plus, [zeros(n,1) Ypreds{j}], yc(cv,j));
        RMSECV{j} = sqrt(mean(bsxfun(@minus,yO(:,j),Ypreds{j}).^2));
        subplot(1,size(q,1),j)
        plot(RMSECV{j},'-or');xlabel('Variables');ylabel('RMSECV');title(['Property ' num2str(j)]);
    end
end
