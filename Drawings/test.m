% hL=plot(randn(10,3));   % the solid lines
% hold on
% hD=plot(0.5*randn(10,3),':');  % and the dotted
% % hX=plot(nan(2,5));             dummy that won't show 'cuz of NaN
% % set(hX(4:5),'color','k')       for the solid/dotted line use black
% % set(hX(5),'linestyle',':')     set the last one for "dotted"
% set(hD,'Parent',hL);
% legend(hD,'Blue','Green','location','best');

plot(sin(1:10),'b')
hold on
plot(cos(1:10),'r')
legend({['blue' char(10) 'line'],'red line'})
