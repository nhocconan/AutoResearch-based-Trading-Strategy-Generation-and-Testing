#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h EMA20 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets:
# - Buy when Williams %R crosses above -80 from below (oversold bounce) in uptrend
# - Sell when Williams %R crosses below -20 from above (overbought rejection) in downtrend
# Uses 4h EMA20 for trend filter to avoid counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
# Session filter (08-20 UTC) reduces noise trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 20-period EMA on 4h close for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Uptrend: look for long entries on Williams %R oversold bounce
        if close[i] > ema20_4h_aligned[i] and volume_filter[i]:
            # Enter long when Williams %R crosses above -80 from below
            if williams_r[i] > -80 and williams_r[i-1] <= -80:
                signals[i] = 0.20
                position = 1
            # Exit long when Williams %R crosses below -20 from above (overbought)
            elif position == 1 and williams_r[i] < -20 and williams_r[i-1] >= -20:
                signals[i] = 0.0
                position = 0
        
        # Downtrend: look for short entries on Williams %R overbought rejection
        elif close[i] < ema20_4h_aligned[i] and volume_filter[i]:
            # Enter short when Williams %R crosses below -20 from above
            if williams_r[i] < -20 and williams_r[i-1] >= -20:
                signals[i] = -0.20
                position = -1
            # Exit short when Williams %R crosses above -80 from below (oversold)
            elif position == -1 and williams_r[i] > -80 and williams_r[i-1] <= -80:
                signals[i] = 0.0
                position = 0
        
        # Hold current position or stay flat
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_WilliamsR_4hEMA20_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0