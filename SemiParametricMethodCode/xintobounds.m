function [x] = xintobounds(x, lbounds, ubounds)
%
% x can be a column vector or a matrix consisting of column vectors
%
if ~isempty(lbounds)
    if length(lbounds) == 1
        idx = x < lbounds;
%         x(idx) = lbounds;
        x(idx) = 2*(lbounds-x(idx));
    else
        arbounds = repmat(lbounds, size(x,1), 1);
        arrbounds = repmat(ubounds, size(x,1), 1);
        idx = x < arbounds;
        if any(idx)
           a = 1;
        end
        %Set to bounds
        x(idx) = arbounds(idx);
        %reflect into bounds
%         x(idx) = x(idx)+(2*(arbounds(idx)-x(idx)));
        % Now double check if all elements are within bounds
        idx3 = x < arbounds;
        x(idx3) = arbounds(idx3);
%         x(idx3) = arbounds(idx3) + rand*(arrbounds(idx3)-arbounds(idx3));
    end
end
if ~isempty(ubounds)
    if length(ubounds) == 1
        idx2 = x > ubounds;
        %x(idx2) = ubounds;
        x(idx2) = 2*(ubounds-x(idx2));
    else
        arbounds = repmat(lbounds, size(x,1), 1);
        arrbounds = repmat(ubounds, size(x,1), 1);
        idx2 = x > arrbounds;
        if any(idx)
           a = 1;
        end
        %Set to bounds
        x(idx2) = arrbounds(idx2);
        %Reflect in to bounds
%         x(idx2) = x(idx2)- (2*(x(idx2)-arrbounds(idx2)));
        % Now double check if all elements are within bounds
        idx4 = x > arrbounds;
        %Set to bounds 
        x(idx4) = arrbounds(idx4);
%         x(idx4) = arbounds(idx4) + rand*(arrbounds(idx4)-arbounds(idx4));
    end
end
if any(x(:) < 0)
   gg = 1;
end
idx = idx2-idx;