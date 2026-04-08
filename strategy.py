#!/usr/bin/env python3
# 1d_1w_1d_volume_breakout_v1
# Hypothesis: Trade daily breakouts above/below previous day's high/low with weekly trend filter and volume confirmation.
# In weekly uptrend: go long when price breaks above previous day's high with volume > 2x average.
# In weekly downtrend: go short when price breaks below previous day's low with volume > 2x average.
# Exit when price returns to previous day's close or weekly trend reverses.
# Uses weekly EMA21 for trend, daily breakout levels for entry, and volume surge filter.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_1d_volume_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA21 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily breakout levels: previous day's high and low
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(prev_high[i]) or 
            np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price returns to previous day's close or weekly trend breaks down
            if close[i] <= prev_close[i] or close[i] < ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to previous day's close or weekly trend breaks up
            if close[i] >= prev_close[i] or close[i] > ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above previous day's high with volume surge and weekly uptrend
            if (high[i] > prev_high[i] and vol_surge and 
                close[i] > ema21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below previous day's low with volume surge and weekly downtrend
            elif (low[i] < prev_low[i] and vol_surge and 
                  close[i] < ema21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals