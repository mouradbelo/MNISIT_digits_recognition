function clickDownGui(object,~,h,X,y,all_theta,widthDigit,heightDigit,rp)

    global idDigit;
    
    if strcmp(get(object,'selectiontype'),'normal')        
        idDigit = idDigit+1;    
        m = size(X,1);
        if (idDigit>m)
            idDigit=m;
        end
    end
    
    if strcmp(get(object,'selectiontype'),'alt')
        idDigit = idDigit-1;    
        if (idDigit<1)
            idDigit=1;
        end                
    end
    
    %-- If middle or right button pressed, then close figure
    if strcmp(get(object,'selectiontype'),'extend')
        set (gcf, 'WindowButtonMotionFcn', []);
        close(h);
        return;
    end


    %-- Ramdomly select one image
    ind = rp(idDigit);
    img = reshape(X(ind,2:end),[heightDigit widthDigit]);
    
    
    %-- Predict digit thanks to logistic regression classifier
    pred = lrc.predict(all_theta, X(ind,:));
    ref = y(ind);
    
    
    %-- Display result
    figure(h); imagesc(img); axis image; colormap(gray); title('Press left/right to navigate / middle to quit'); axis off;
    if (pred == ref)
        text(2,3,{['True value = ',num2str(ref)],['Prediction = ',num2str(pred)]},'Color','green','FontSize',10);
    else
        text(2,3,{['True value = ',num2str(ref)],['Prediction = ',num2str(pred)]},'Color','red','FontSize',10);
    end

end
