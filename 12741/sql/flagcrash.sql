SELECT  flag.*,crash.MAX_SEVERITY_LEVEL,crash.URBAN_RURAL,crash.ILLUMINATION,crash.WEATHER
FROM test.flag
join test.crash
on crash.CRN=flag.CRN;