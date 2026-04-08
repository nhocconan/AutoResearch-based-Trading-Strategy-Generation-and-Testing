#!/usr/bin/env python3
# 1d_1w_trend_ema_volume_v1
# Hypothesis: Trade with weekly trend using daily EMA20 for entries and volume confirmation.
# In weekly uptrend: go long when price closes above EMA20 with volume > 2x average.
# In weekly downtrend: go short when price closes below EMA20 with volume > 2x average.
# Exit when price crosses back over EMA20 or weekly trend changes.
# Uses weekly EMA for trend filter and daily EMA20 for entry timing.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_trend_ema_volume_v1"
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
    
    # Daily EMA20 for entry signal
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30  # Ensure EMA20 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(ema20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price < EMA20 or weekly trend turns down
            if close[i] < ema20[i] or close[i] < ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > EMA20 or weekly trend turns up
            if close[i] > ema20[i] or close[i] > ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > EMA20 with volume surge and weekly uptrend
            if (close[i] > ema20[i] and vol_surge and 
                close[i] > ema21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < EMA20 with volume surge and weekly downtrend
            elif (close[i] < ema20[i] and vol_surge and 
                  close[i] < ema21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals