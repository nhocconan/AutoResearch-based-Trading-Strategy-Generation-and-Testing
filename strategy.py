#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Williams %R with 1-week EMA trend filter and volume confirmation.
# Williams %R measures overbought/oversold conditions on a 0 to -100 scale.
# Readings below -80 indicate oversold (potential long), above -20 indicate overbought (potential short).
# In ranging markets, these extremes often precede reversals.
# Trend filter: 1-week EMA determines primary trend direction - only take longs in uptrend, shorts in downtrend.
# Volume confirmation ensures institutional participation.
# Designed for ~10-25 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    
    williams_r = ((highest_high - close) / hh_ll) * -100
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 34-period EMA on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals with trend filter
        # Oversold condition for long: %R < -80
        # Overbought condition for short: %R > -20
        if williams_r[i] < -80:  # Oversold - potential long
            if close[i] > ema34_1w_aligned[i] and volume_filter[i]:  # Only long in uptrend
                signals[i] = 0.25
                position = 1
        elif williams_r[i] > -20:  # Overbought - potential short
            if close[i] < ema34_1w_aligned[i] and volume_filter[i]:  # Only short in downtrend
                signals[i] = -0.25
                position = -1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WilliamsR_1wEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0