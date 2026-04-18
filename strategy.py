# The key insight from repeated failures: too many trades, not too few.
# We need EXTREMELY tight entry conditions to stay under the 400-trade limit for 4h.
# Hypothesis: Only trade when price breaks a 4-hour Donchian channel AND
# daily volatility is in the top 20% (avoid chop) AND volume is 2x average.
# This should yield ~15-25 trades/year, well under the limit.
# Works in bull markets (breakouts up) and bear markets (breakouts down).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Volatile_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian channels (20-period) - use previous period's data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Daily True Range and ATR (14-period Wilder's smoothing)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = atr[i-1] * (1 - 1/atr_period) + tr[i] * (1/atr_period)
            else:
                atr[i] = np.nan
    
    # Align ATR to 4h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate ATR percentile over last 50 days (avoid look-ahead)
    atr_percentile = np.full_like(atr_aligned, np.nan)
    lookback = 50  # days
    for i in range(lookback, len(atr_aligned)):
        window = atr_aligned[i-lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            current = atr_aligned[i]
            if not np.isnan(current):
                percentile = (np.sum(valid < current) / len(valid)) * 100
                atr_percentile[i] = percentile
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Session filter: 08-20 UTC (most liquid hours)
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(60, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any data missing
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_percentile[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Extreme conditions: volatility in top 20% AND volume 2x average
        volatile = atr_percentile[i] >= 80
        vol_surge = volume[i] >= (vol_ma_20[i] * 2.0)
        
        if position == 0:
            # Only enter on breakout with extreme conditions
            if volatile and vol_surge:
                if close[i] > high_20[i]:
                    signals[i] = 0.30
                    position = 1
                elif close[i] < low_20[i]:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Exit: volatility drops OR price retrace to midpoint
            midpoint = (high_20[i] + low_20[i]) / 2
            if atr_percentile[i] < 50 or close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: volatility drops OR price retrace to midpoint
            midpoint = (high_20[i] + low_20[i]) / 2
            if atr_percentile[i] < 50 or close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals