#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d trend filter (EMA50)
# Donchian breakouts capture momentum bursts; volume confirmation ensures institutional participation
# 1d EMA50 trend filter avoids counter-trend trades in choppy markets
# Designed for 6h timeframe to balance trade frequency and signal quality
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_Donchian20_12hVolume_1dEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume MA(20) for confirmation
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian calculation)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper + volume > 1.5x 12h MA + close > 1d EMA50
            if (close[i] > high_roll_max[i] and 
                volume[i] > (vol_ma_12h_aligned[i] * 1.5) and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + volume > 1.5x 12h MA + close < 1d EMA50
            elif (close[i] < low_roll_min[i] and 
                  volume[i] > (vol_ma_12h_aligned[i] * 1.5) and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower (reversal signal)
            if close[i] < low_roll_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper (reversal signal)
            if close[i] > high_roll_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals