#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w EMA50 trend filter
# Long when: price breaks above 20-period Donchian high AND 1d volume > 2.0x 20-period average AND 1w close > 1w EMA50
# Short when: price breaks below 20-period Donchian low AND 1d volume > 2.0x 20-period average AND 1w close < 1w EMA50
# Exit when price returns to 20-period Donchian middle (mean reversion in range)
# Donchian breakout captures structural moves, volume spike confirms participation, 1w EMA50 ensures higher timeframe alignment
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25 to minimize fee churn

name = "4h_Donchian20_1dVolumeSpike_1wEMA50"
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
    
    # Get 1d data ONCE before loop for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume average
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1d average volume (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian Channels (20) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high + volume spike + 1w uptrend
            if (close[i] > highest_20[i] and 
                volume[i] > 2.0 * vol_ma_aligned[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume spike + 1w downtrend
            elif (close[i] < lowest_20[i] and 
                  volume[i] > 2.0 * vol_ma_aligned[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to Donchian middle (mean reversion)
            if close[i] < middle_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to Donchian middle (mean reversion)
            if close[i] > middle_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals