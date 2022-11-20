use test;
CREATE TABLE if NOT exists Crash (
    CRN varchar(255) NOT NULL,
    COLLISION_TYPE INT NOT NULL,
	CRASH_time INT NOT NULL,
	DAY_OF_WEEK INT NOT NULL,
    DEC_LAT varchar(255) NOT NULL,
    DEC_LONG varchar(255) NOT NULL,
    FATAL_COUNT INT NOT NULL,
    HOUR_OF_DAY INT NOT NULL,
    ILLUMINATION INT NOT NULL,
    INJURY_COUNT INT NOT NULL,
    MAX_SEVERITY_LEVEL INT NOT NULL,
    TIME_OF_DAY INT NOT NULL,
    WEATHER INT NOT NULL,
    VEHICLE_COUNT INT NOT NULL,
    URBAN_RURAL INT NOT NULL,
    PRIMARY KEY (CRN)
);

CREATE TABLE if NOT exists flag(
     CRN varchar(255) NOT NULL,
     CRASH_time INT NOT NULL,
     WET_ROAD INT NOT NULL, 
     SNOW_SLUSH_ROAD INT NOT NULL,
     ICY_ROAD INT NOT NULL,
     ALCOHOL_RELATED INT NOT NULL,
     DRINKING_DRIVER INT NOT NULL,
     UNLICENSED INT NOT NULL,
	 CELL_PHONE INT NOT NULL,
	 SPEEDING_RELATED INT NOT NULL,
     AGGRESSIVE_DRIVING INT NOT NULL,
     FATIGUE_ASLEEP INT NOT NULL,
     PRIMARY KEY (CRN),
     FOREIGN KEY(CRN) references Crash(CRN)
);

CREATE TABLE if NOT exists ROAD (
    CRN varchar(255) NOT NULL,
    SequenceID INT NOT NULL,
    Lanecount varchar(255) NOT NULL,
    SpeedLimit varchar(255) NOT NULL,
    PRIMARY KEY (CRN,SequenceID),
    FOREIGN KEY(CRN) references Crash(CRN)
);

CREATE TABLE  if NOT exists Pixer(
    PixerID varchar(255) NOT NULL,
    Latitude varchar(255) NOT NULL,
    Longitude varchar(255) NOT NULL,
    primary key (PixerID)
);

CREATE TABLE if NOT exists rainfall(
    PixerID varchar(255) NOT NULL,
    time varchar(255) NOT NULL,
    volume INT NOT NULL,
    FOREIGN KEY(PixerID) references Pixer(PixerID),
    primary key(time,PixerID)
);

CREATE TABLE if NOT exists nearby(
    PixerID varchar(255) NOT NULL,
    CRN varchar(255) NOT NULL,
    FOREIGN KEY(PixerID) references Pixer(PixerID),
    FOREIGN KEY(CRN) references Crash(CRN),
    primary key(PixerID,CRN)
    );




