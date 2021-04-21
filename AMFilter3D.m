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
Ns=5; % !!!!! WATCH OUT HOW Ns IS USED !!!!!!!!!
Q=P+log(Ns)/log(xi_0); 
SHIFT = 100*realmin^(1/P); % small shift to prevent division by 0
BACKSHIFT = 0.95*Ns^(1/Q)*SHIFT^(P/Q);
Xi=zeros(size(x)); keep=zeros(size(x)); sq=zeros(size(x));
% baseline: identity
xi(nely,:,:)=x(nely,:,:); % copy base row as-is
for i=(nely-1):-1:1
    % compute maxima of current base row
%     cbr = [0, xi(i+1,:), 0] + SHIFT; % pad with zeros
    cbr = zeros(1,nelx+2,nelz+2);
    cbr(1,2:nelx+1, 2:nelz+1) = xi(i+1,:,:)+SHIFT;
    
%     keep(i,:) = (cbr(1:nelx).^P + cbr(2:(nelx+1)).^P + cbr(3:end).^P);

%     keep1 = (cbr(:,1:nelx).^P + cbr(:, 2:(nelx+1)).^P + cbr(:, 3:end).^P);
%     keep2 = (cbr(1:nelz,:).^P + cbr(2:(nelz+1),:).^P + cbr(3:end, :).^P);
%     keep(i,:,:) = max(keep1,keep2);
    keep(i,:,:) = (cbr(1,1:nelx,2:(nelz+1)).^P + cbr(1,2:(nelx+1),2:(nelz+1)).^P + cbr(1,3:end,2:(nelz+1)).^P ...
                  +cbr(1,2:(nelx+1),1:nelz).^P + cbr(1,2:(nelx+1),3:end).^P);
    
    Xi(i,:,:) = keep(i,:,:).^(1/Q) - BACKSHIFT;
    sq(i,:,:) = sqrt((x(i,:,:)-Xi(i,:,:)).^2 + ep);
    % set row above to supported value using smooth minimum:
    xi(i,:,:) = 0.5*((x(i,:,:)+Xi(i,:,:)) - sq(i,:,:) + sqrt(ep));
end
%SENSITIVITIES
if nSens
    dfxi=varargin; dfx=varargin; 
%     lambda=zeros(nSens,nelx); % MAYBE MAKE IT 3D (zeros(nelz,nelx, nSens))?
    lambda = zeros(nSens, nelx,nelz);
    % from top to base layer:
    for i=1:nely-1
        % smin sensitivity terms
        dsmindx  = .5*(1-(x(i,:,:)-Xi(i,:,:))./sq(i,:,:));
        %dsmindXi = .5*(1+(x(i,:)-Xi(i,:))./sq(i,:)); 
        dsmindXi = 1-dsmindx; 
        % smax sensitivity terms
%         cbr = [0, xi(i+1,:), 0] + SHIFT; % pad with zeros
        cbr = zeros(1,nelx+2,nelz+2);
        cbr(1,2:nelx+1, 2:nelz+1) = xi(i+1,:,:)+SHIFT;
        
%         dmx = zeros(Ns,nelx);
%         for j=1:Ns
%             dmx(j,:) = (P/Q)*keep(i,:).^(1/Q-1).*cbr((1:nelx)+(j-1)).^(P-1);
%         end        
        dmx = zeros(Ns, nelx, nelz);
        for j = 1:Ns
            if j <= 3
                dmx(j,:,:) = (P/Q)*keep(i,:,:).^(1/Q-1).*cbr(1,(1:nelx)+(j-1),(2:nelz+1)).^(P-1);
            elseif j == 4
                dmx(j,:,:) = (P/Q)*keep(i,:,:).^(1/Q-1).*cbr(1,(2:nelx+1),(1:nelz)).^(P-1);
            elseif j == 5
                dmx(j,:,:) = (P/Q)*keep(i,:,:).^(1/Q-1).*cbr(1,(2:nelx+1),(1:nelz)+2).^(P-1);
            end
        end
        
%         % rearrange data for quick multiplication:
%         qj=repmat([-1 0 1]',nelx,1);
%         qi=repmat(1:nelx,3,1); qi=qi(:);
%         qj=qj+qi; qs=dmx(:);
%         dsmaxdxi=sparse(qi(2:end-1),qj(2:end-1),qs(2:end-1)); 
        
        for k=1:nSens
            dfx{k}(i,:,:) = dsmindx.*(dfxi{k}(i,:,:)+lambda(k,:,:));
%              lambda(k,:)= ((dfxi{k}(i,:)+lambda(k,:)).*dsmindXi)*dsmaxdxi;
            preLambda = zeros(1,nelx+2,nelz+2);
            preLambda(1,2:nelx+1,2:nelz+1) = (dfxi{k}(i,:,:)+lambda(k,:,:)).*dsmindXi;
            for nz=1:nelz
                for nx = 1:nelx
                    lambda(k, nx, nz) = sum([preLambda(1, nx-1+1,nz+1);...
                                            preLambda(1, nx+1,nz+1);...
                                            preLambda(1, nx+1+1,nz+1);...
                                            preLambda(1, nx+1,nz-1+1);...
                                            preLambda(1, nx+1,nz+1+1)].*...
                                            dmx(2,nx,nz));
%    [dmx(1,nx,nz);dmx(2,nx,nz); dmx(3,nx,nz);dmx(4,nx,nz); dmx(5,nx,nz)]
                                            
                end
            end
        end
    end
    % base layer:
    i=nely;
    for k=1:nSens
        dfx{k}(i,:,:) = dfxi{k}(i,:,:)+lambda(k,:,:);
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