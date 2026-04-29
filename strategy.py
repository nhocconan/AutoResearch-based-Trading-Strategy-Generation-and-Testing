#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian upper channel AND 12h EMA50 uptrend AND volume spike
# Short when price breaks below Donchian lower channel AND 12h EMA50 downtrend AND volume spike
# Exit on opposite Donchian break or trend reversal
# Uses discrete position sizing (0.25) to minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Donchian_Breakout_12hEMA50_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # max(20, 50) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower OR 12h EMA50 turns down (price < EMA)
            if curr_low < lowest_low_20[i] or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper OR 12h EMA50 turns up (price > EMA)
            if curr_high > highest_high_20[i] or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper AND price above 12h EMA50 AND volume spike
            if (curr_high > highest_high_20[i] and 
                curr_close > curr_ema_12h and
                volume[i] > 2.0 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower AND price below 12h EMA50 AND volume spike
            elif (curr_low < lowest_low_20[i] and 
                  curr_close < curr_ema_12h and
                  volume[i] > 2.0 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals