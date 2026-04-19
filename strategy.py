#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout (20-period) with 1d EMA200 trend filter and volume confirmation.
# Long when: price breaks above Donchian upper (20-period high) and close > 1d EMA200 and volume > 1.8x 20-period average
# Short when: price breaks below Donchian lower (20-period low) and close < 1d EMA200 and volume > 1.8x 20-period average
# Exit when: price crosses back inside Donchian channel (upper for long exit, lower for short exit)
# Uses 12h timeframe for entry timing with 1d trend filter to reduce whipsaw in bear markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 12-25 trades/year per symbol (50-100 total over 4 years).
name = "12h_Donchian_Breakout_EMA200_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA200 for trend filter (loaded once, aligned)
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 calculation (longest indicator)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: break above upper band, above EMA200, volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema200_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower band, below EMA200, volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema200_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back inside Donchian (below upper band)
            if close[i] < high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back inside Donchian (above lower band)
            if close[i] > low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals