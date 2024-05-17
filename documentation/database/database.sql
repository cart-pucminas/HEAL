CREATE USER userWhoPrediction
CREATE ROLE userWhoPrediction SUPERUSER;
ALTER ROLE userWhoPrediction PASSWORD 'prmTW(fWY75JDgq!qqg(VnvCBfgpLmdKnbSq1CF';
ALTER ROLE userWhoPrediction LOGIN;
CREATE DATABASE databaseWhoPrediction;
ALTER DATABASE databaseWhoPrediction OWNER to userWhoPrediction;



CREATE TABLE regions (
    region_id VARCHAR(36) NOT NULL,
    name VARCHAR(100) NOT NULL,
    code VARCHAR(10) NOT NULL,
    PRIMARY KEY (region_id)
);

CREATE TABLE countries (
    country_id VARCHAR(36) NOT NULL,
    name VARCHAR(100) NOT NULL,
    code VARCHAR(20) NOT NULL,
    region_id VARCHAR(36) NOT NULL,
    PRIMARY KEY (country_id),
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

CREATE TABLE years (
    year_id VARCHAR(36) NOT NULL,
    year VARCHAR(30) NOT NULL,
    code VARCHAR(30) NOT NULL,
    PRIMARY KEY (year_id)
);

CREATE TABLE indicators (
    indicator_id VARCHAR(36) NOT NULL,
    name VARCHAR(300) NOT NULL,
    code VARCHAR(30) NOT NULL,
    PRIMARY KEY (indicator_id)
);

CREATE TABLE dimensions (
    dimension_id VARCHAR(36) NOT NULL,
    name VARCHAR(100) NOT NULL,
    code VARCHAR(100) NOT NULL,
    original_dimension VARCHAR(100) NOT NULL,
    PRIMARY KEY (dimension_id)
);

CREATE TABLE datas (
    data_id VARCHAR(36) NOT NULL,
    value text NOT NULL,
    country_id VARCHAR(36),
    year_id VARCHAR(36),
    indicator_id VARCHAR(36),
    PRIMARY KEY (data_id),
    FOREIGN KEY (country_id) REFERENCES countries(country_id),
    FOREIGN KEY (indicator_id) REFERENCES indicators(indicator_id),
    FOREIGN KEY (year_id) REFERENCES years(year_id)
);

CREATE TABLE datas_dimensions (
    data_dimension_id VARCHAR(36) NOT NULL,
    data_id VARCHAR(36) NOT NULL,
    dimension_id VARCHAR(36) NOT NULL,
    PRIMARY KEY (data_dimension_id),
    FOREIGN KEY (data_id) REFERENCES datas(data_id),
    FOREIGN KEY (dimension_id) REFERENCES dimensions(dimension_id)
);

ALTER TABLE regions owner to userWhoPrediction;
ALTER TABLE countries owner to userWhoPrediction;
ALTER TABLE years owner to userWhoPrediction;
ALTER TABLE dimensions owner to userWhoPrediction;
ALTER TABLE datas owner to userWhoPrediction;
ALTER TABLE datas_dimensions owner to userWhoPrediction;
ALTER TABLE indicators owner to userWhoPrediction;