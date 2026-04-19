# %%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RM_1D_breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # 1d rolling max/min for breakout levels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling max and min with proper min_periods
    roll_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    roll_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    roll_max_20_aligned = align_htf_to_ltf(prices, df_1d, roll_max_20)
    roll_min_20_aligned = align_htf_to_ltf(prices, df_1d, roll_min_20)
    
    # 1d ATR for volatility filter (14-period)
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(roll_max_20_aligned[i]) or np.isnan(roll_min_20_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above 20-period high
            if price > roll_max_20_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 20-period low
            elif price < roll_min_20_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below 20-period low or ATR-based stop
            if price < roll_min_20_aligned[i] or price < np.max(high[i-4:i+1]) - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above 20-period high or ATR-based stop
            if price > roll_max_20_aligned[i] or price > np.min(low[i-4:i+1]) + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%