#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA trend filter and volume spike.
# Williams %R measures overbought/oversold conditions.
# Williams %R < -80 = oversold (long opportunity)
# Williams %R > -20 = overbought (short opportunity)
# Strategy: Enter long when Williams %R < -80 in uptrend (price > 12h EMA50) with volume spike
# Enter short when Williams %R > -20 in downtrend (price < 12h EMA50) with volume spike
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Designed for ~15-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    willr = -100 * ((highest_high - close) / hl_range)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Williams %R oversold + uptrend + volume
        if willr[i] < -80 and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short conditions: Williams %R overbought + downtrend + volume
        elif willr[i] > -20 and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit long: Williams %R crosses above -50
        elif position == 1 and willr[i] > -50:
            signals[i] = 0.0
            position = 0
        # Exit short: Williams %R crosses below -50
        elif position == -1 and willr[i] < -50:
            signals[i] = 0.0
            position = 0
        # Hold current position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_12hEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0