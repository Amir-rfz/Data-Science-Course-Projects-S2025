UPDATE flights f
JOIN airlines a ON LOWER(TRIM(f.AIRLINE)) = LOWER(TRIM(a.IATA_CODE))
SET f.airline_id = a.airline_id;

UPDATE flights f1
JOIN airports a1 ON LOWER(TRIM(f1.ORIGIN_AIRPORT)) = LOWER(TRIM(a1.IATA_CODE))
SET f1.origin_airport_id = a1.airport_id;

UPDATE flights f2
JOIN airports a1 ON LOWER(TRIM(f2.ORIGIN_AIRPORT)) = LOWER(TRIM(a2.IATA_CODE))
SET f2.origin_airport_id = a2.airport_id;


