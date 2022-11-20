SELECT crash.CRN,crash.MAX_SEVERITY_LEVEL,crash.URBAN_RURAL,crash.ILLUMINATION,crash.WEATHER,road.Lanecount,road.SpeedLimit
FROM test.road
inner join test.crash
on crash.CRN=road.CRN;