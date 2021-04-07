function [ xi, varargout ] = AMFilter3D( x, baseplate, varargin )
%AMFILTER Applies a virtual additive manufacturing process to a 
%         2D blueprint design input.
%   Possible uses:
%   xi = AMfilter(x)              design transformation, default orientation
%   xi = AMfilter(x, baseplate)   idem, with baseplate orientation specified
%   [xi, df1dx, df2dx,...] = AMfilter(x, baseplate, df1dxi, df2dxi, ...)
%       This includes also the transformation of design sensitivities
% where
%   x : blueprint design (2D array), 0 <= x(i,j) <= 1
%   xi: printed design (2D array)
%   baseplate: character indicating baseplate orientation: 'N','E','S','W'
%              default orientation is 'S'
%              for 'X', the filter is inactive and just returns the input.
%   df1dx, df1dxi etc.:  design sensitivity (2D arrays)

%INTERNAL SETTINGS
P = 40; ep = 1e-4; xi_0 = 0.5; % parameters for smooth max/min functions

%INPUT CHECKS
if nargin==1, baseplate='S'; end 
if baseplate=='X', 
    % bypass option: filter does not modify the blueprint design
    xi = x;
    varargout = varargin;
    return;
end 
nRot=find(upper(baseplate)=='SWNE')-1;
nSens=max(0,nargin-2); 
if nargout~=nSens+1, error('Input/output arguments mismatch.'); end

%ORIENTATION
x=rot90(x,nRot);
xi=zeros(size(x));
for s=1:nSens
    varargin{s}=rot90(varargin{s},nRot);    
end
[nely,nelx, nelz]=size(x); 

%AM FILTER =====================
Ns=3;
Q=P+log(Ns)/log(xi_0); 
SHIFT = 100*realmin^(1/P); % small shift to prevent division by 0
BACKSHIFT = 0.95*Ns^(1/Q)*SHIFT^(P/Q);
XiX=zeros(size(x)); keepX=zeros(size(x)); sqX=zeros(size(x));
XiY=zeros(size(x)); keepY=zeros(size(x)); sqY=zeros(size(x));
% baseline: identity
xi(:,:,nelz)=x(:,:,nelz); % copy base face as-is
xiY = zeros(size(x));
xiX = zeros(size(x));

for i=(nelz-1):-1:1
    for j = 1:nely
    % compute maxima of current base row
    cbr = [0, xi(j,:,i+1), 0] + SHIFT; % pad with zeros
    keepY(j,:,i) = (cbr(1:nelx).^P + cbr(2:(nelx+1)).^P + cbr(3:end).^P);
    XiY(j,:,i) = keepY(j,:,i).^(1/Q) - BACKSHIFT;
    sqY(j,:,i) = sqrt((x(j,:,i)-XiY(j,:,i)).^2 + ep);
    % set row above to supported value using smooth minimum:
    xiY(j,:,i) = 0.5*((x(j,:,i)+XiY(j,:,i)) - sqY(j,:,i) + sqrt(ep));
    end
    
    for j = 1:nelx
    % compute maxima of current base row
    cbr = [0; xi(:,j,i+1); 0] + SHIFT; % pad with zeros
    keepX(:,j,i) = (cbr(1:nely).^P + cbr(2:(nely+1)).^P + cbr(3:end).^P);
    XiX(:,j,i) = keepX(:,j,i).^(1/Q) - BACKSHIFT;
    sqX(:,j,i) = sqrt((x(:,j,i)-XiX(:,j,i)).^2 + ep);
    % set row above to supported value using smooth minimum:
    xiX(:,j,i) = 0.5*((x(:,j,i)+XiX(:,j,i)) - sqX(:,j,i) + sqrt(ep));
    end
end

%SENSITIVITIES
if nSens
    dfxiCol=varargin; dfxCol=varargin;
    dfxiRow=varargin; dfxRow=varargin;
    
    for j = 1:nelx
        lambdaRow=zeros(nSens,nely); 
        % from top to base layer:
        for i=1:nelz-1
            % smin sensitivity terms
            dsmindx  = .5*(1-(x(:,j,i)-XiX(:,j,i))./sqX(:,j,i));
            %dsmindXi = .5*(1+(x(i,:)-Xi(i,:))./sq(i,:)); 
            dsmindXi = 1-dsmindx; 
            % smax sensitivity terms
            cbr = [0; xi(:,j,i+1); 0] + SHIFT; % pad with zeros
            dmx = zeros(Ns,nelx);
            for s=1:Ns
                dmx(s,:) = (P/Q)*keepX(:,j,i).^(1/Q-1).*cbr((1:nely)+(s-1)).^(P-1);
            end        
            % rearrange data for quick multiplication:
            qj=repmat([-1 0 1]',nely,1);
            qi=repmat(1:nely,3,1); qi=qi(:);
            qj=qj+qi; qs=dmx(:);
            dsmaxdxi=sparse(qi(2:end-1),qj(2:end-1),qs(2:end-1)); 
            for k=1:nSens
                dfxRow{k}(:,j,i) = dsmindx.*(dfxiRow{k}(:,j,i)+lambdaRow(k,:)');
                lambdaRow(k,:)= ((dfxiRow{k}(:,j,i)+lambdaRow(k,:)').*dsmindXi)'*dsmaxdxi;
            end
        end
        % base layer:
        i=nelz;
        for k=1:nSens
            dfxRow{k}(:,j,i) = dfxiCol{k}(:,j,i)+lambdaRow(k,:)';
        end
    end
    
    for j = 1:nely
        lambdaCol=zeros(nSens,nelx); 
        % from top to base layer:
        for i=1:nelz-1
            % smin sensitivity terms
            dsmindx  = .5*(1-(x(j,:,i)-XiX(j,:,i))./sqX(j,:,i));
            %dsmindXi = .5*(1+(x(i,:)-Xi(i,:))./sq(i,:)); 
            dsmindXi = 1-dsmindx; 
            % smax sensitivity terms
            cbr = [0, xi(j,:,i+1), 0] + SHIFT; % pad with zeros
            dmx = zeros(Ns,nely);
            for s=1:Ns
                dmx(s,:) = (P/Q)*keepY(j,:,i).^(1/Q-1).*cbr((1:nelx)+(s-1)).^(P-1);
            end        
            % rearrange data for quick multiplication:
            qj=repmat([-1 0 1]',nelx,1);
            qi=repmat(1:nelx,3,1); qi=qi(:);
            qj=qj+qi; qs=dmx(:);
            dsmaxdxi=sparse(qi(2:end-1),qj(2:end-1),qs(2:end-1)); 
            for k=1:nSens
                dfxCol{k}(j,:,i) = dsmindx.*(dfxiCol{k}(j,:,i)+lambdaCol(k,:));
                lambdaCol(k,:)= ((dfxiCol{k}(j,:,i)+lambdaCol(k,:)).*dsmindXi)*dsmaxdxi;
            end
        end
        % base layer:
        i=nelz;
        for k=1:nSens
            dfxCol{k}(:,j,i) = dfxCol{k}(:,j,i)+lambdaCol(k,:)';
        end
    end
    
    for k = 1:nSens
        dfx{k} = 0.5*(dfxCol{k}+dfxRow{k});
    end
end

%ORIENTATION
xi=rot90(xi,-nRot);
for s=1:nSens
    varargout{s}=rot90(dfx{s},-nRot);    
end

end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Matlab code was written by Matthijs Langelaar,                      %
% Department of Precision and Microsystems Engineering,                    %
% Delft University of Technology, Delft, the Netherlands.                  %
% Please sent your comments to: m.langelaar@tudelft.nl                     %
%                                                                          %
% The code is intended for educational purposes and theoretical details    %
% are discussed in the paper "An additive manufacturing filter for         %
% topology optimization of print-ready designs", M. Langelaar (2016),      %
% Struct Multidisc Optim, DOI: 10.1007/s00158-016-1522-2.                  %
%                                                                          %
% This code is intended for integration in the 88-line topology            %
% optimization code discussed in the paper                                 %
% "Efficient topology optimization in MATLAB using 88 lines of code,       %
% E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov and O. Sigmund,  % 
% Struct Multidisc Optim, 2010, Vol 21, pp. 120--127.                      %
%                                                                          %
% Disclaimer:                                                              %
% The author reserves all rights but does not guarantee that the code is   %
% free from errors. Furthermore, the author shall not be liable in any     %
% event caused by the use of the program.                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%