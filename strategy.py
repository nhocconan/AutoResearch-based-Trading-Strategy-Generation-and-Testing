#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation
# Donchian channels provide clear breakout levels that capture momentum in both trending and ranging markets
# 12h EMA34 ensures trades align with higher-timeframe trend to reduce false signals
# Volume confirmation at 1.8x average filters low-participation moves
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Discrete sizing 0.25 to balance profit potential and fee drag

name = "4h_Donchian20_Breakout_12hEMA34_Volume"
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
    
    # Calculate 12h Donchian(20) channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian channels: 20-period high/low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper channel: highest high of last 20 periods
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h channels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: Price breaks above upper channel AND price > 12h EMA34 AND volume spike
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: Price breaks below lower channel AND price < 12h EMA34 AND volume spike
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below lower channel (breakdown) OR closes below 12h EMA34 (trend change)
            if close[i] < lower_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above upper channel (breakout) OR closes above 12h EMA34 (trend change)
            if close[i] > upper_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals