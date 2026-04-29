#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter
# Uses Donchian channel breakouts from 20-period high/low for trend continuation
# Volume confirmation (>1.5x 20-period average) filters false breakouts
# 1d EMA50 trend filter ensures alignment with higher timeframe momentum
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
# Proven pattern: Donchian + volume + trend = SOLUSDT test Sharpe 1.10-1.38 (from research)

name = "4h_Donchian20_VolumeConfirmation_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) from 4h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 20)  # Donchian, 1d EMA, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = high_roll[i]
        curr_donchian_low = low_roll[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits and trailing logic
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low
            if curr_low < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high
            if curr_high > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high with volume confirmation and uptrend
            if vol_confirm and curr_high > curr_donchian_high and curr_close > curr_ema_1d:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume confirmation and downtrend
            elif vol_confirm and curr_low < curr_donchian_low and curr_close < curr_ema_1d:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals