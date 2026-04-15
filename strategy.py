# 6h_12h_Donchian_1d_Volume_CCI - Novel combination using 12h Donchian breakout with 1d volume and CCI confirmation
# Works in bull markets (breakouts up) and bear markets (breakouts down) with volume confirmation to filter false breakouts
# CCI helps identify overbought/oversold conditions to avoid chasing extended moves
# Target: 50-150 total trades over 4 years = 12-37/year

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian breakout
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for volume and CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian Channel (20-period) on 12h
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate CCI (20-period) on 1d
    typical_price = (high_1d + low_1d + close_1d) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
    
    # Align CCI to 6h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / (vol_ma + 1e-10)
    
    # Align volume ratio to 6h timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(cci_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            continue
        
        # Long entry: price breaks above 12h Donchian high + volume confirmation + CCI not overbought
        if (close[i] > donchian_high_aligned[i] and
            vol_ratio_aligned[i] > 1.5 and
            cci_aligned[i] < 100 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below 12h Donchian low + volume confirmation + CCI not oversold
        elif (close[i] < donchian_low_aligned[i] and
              vol_ratio_aligned[i] > 1.5 and
              cci_aligned[i] > -100 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite breakout or CCI reaches extreme levels
        elif position == 1 and (close[i] < donchian_low_aligned[i] or cci_aligned[i] > 200):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high_aligned[i] or cci_aligned[i] < -200):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_12h_Donchian_1d_Volume_CCI"
timeframe = "6h"
leverage = 1.0