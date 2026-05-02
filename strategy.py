#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian channels provide robust structure-based breakouts that work in both trending and ranging markets
# 1d EMA50 ensures trades align with intermediate-term trend to reduce false signals
# Volume confirmation at 2.5x average filters low-participation moves
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Discrete sizing 0.25 to balance profit potential and fee drag

name = "12h_Donchian20_Breakout_1dEMA50_Volume"
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
    
    # Calculate 1d Donchian(20) channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need 20 for Donchian + 1 for previous
        return np.zeros(n)
    
    # Previous day's high/low for Donchian breakout
    prev_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    prev_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 1d levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_20)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: Price breaks above Donchian high AND price > 1d EMA50 AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: Price breaks below Donchian low AND price < 1d EMA50 AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian low (mean reversion) OR closes below 1d EMA50 (trend change)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian high (mean reversion) OR closes above 1d EMA50 (trend change)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals