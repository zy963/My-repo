SELECT crash.CRN, test.crash.MAX_SEVERITY_LEVEL,rainfall.volume,crash.URBAN_RURAL,crash.ILLUMINATION,crash.WEATHER
FROM test.crash,
test.rainfall,
test.pixer,
test.nearby
where rainfall.PixerID=pixer.PixerID
and pixer.PixerID=nearby.PixerID
and nearby.CRN=crash.CRN