function [ ] = plot_param_space( filename, resolution )
    
    A = csvread(filename);
     
    x = reshape(A(:,1),resolution,resolution);
    y = reshape(A(:,2),resolution,resolution);
    z = reshape(A(:,3),resolution,resolution);

    %x = log(x);
    %y = log(y);

    %z = log(z);
    
    figure
    surf(x,y,z)
    title(filename,'interpreter','None')
    xlabel('alpha')
    ylabel('gamma')
    zlabel('bla')
    
    display(strcat('The minimum value for this search is = ', num2str(min(min(z)))))
end

