SELECT crash.CRN,crash.MAX_SEVERITY_LEVEL,rainfall.volume,crash.URBAN_RURAL,crash.ILLUMINATION,crash.WEATHER,
crash.URBAN_RURAL,crash.ILLUMINATION,crash.WEATHER, flag.*,road.Lanecount,road.SpeedLimit
FROM test.crash,
test.rainfall,
test.pixer,
test.road,
test.nearby,
test.flag
where rainfall.PixerID=pixer.PixerID
and pixer.PixerID=nearby.PixerID
and nearby.CRN=crash.CRN
and crash.CRASH_time=rainfall.time
and crash.CRN=flag.CRN
and crash.CRN=road.CRN;