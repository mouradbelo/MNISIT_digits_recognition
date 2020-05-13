function [Xk, gk, i] = gradient_descent(f, Xk, options)

    if (~exist('options','var'))
        options.MaxIter = 50;
        options.epsilon = 0.01;
        options.tau = 1;
    end    
    
    i = 0;    
    [fk,gk] = f(Xk);
    dk = - gk;
       
    %-- Main loop
    while ( (norm(gk)/norm(fk) > options.epsilon) && (i<options.MaxIter) )

        %-- internal iterator
        i = i+1;
        
        Xk = Xk + options.tau*dk;
        [fk,gkp] = f(Xk);
        dk = - gkp;
        gk = gkp;
      
        %-- display iterative cost value
        S = 'Iteration ';
        f1 = norm(fk);
        fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
        
    end
    fprintf('\n');
    
end


