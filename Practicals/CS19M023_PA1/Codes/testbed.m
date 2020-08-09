  arm = 10;
  runs = 2000;
  data = normrnd(0,1,runs,arm);
  boxplot(data);
  xlable(" bandits");
  ylabel("distribution reward");
