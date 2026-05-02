#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian channels identify volatility-based breakouts with clear support/resistance levels
# Breakouts above upper channel or below lower channel with volume indicate strong momentum
# 1d EMA50 provides higher timeframe trend filter to reduce false breakouts in choppy markets
# Volume confirmation (1.8x 24-period average) ensures institutional participation
# Targets 75-200 trades over 4 years (19-50/year) for 4h timeframe as per experiment #117980
# Works in both bull and bear markets by following higher timeframe trend direction

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels (20-period) from 4h data
    # Upper channel = highest high of last 20 periods
    # Lower channel = lowest low of last 20 periods
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume confirmation (1.8x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume calculations)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper channel + price > 1d EMA50 + volume confirmation
            if close[i] > high_roll_max[i] and close[i] > ema_50_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower channel + price < 1d EMA50 + volume confirmation
            elif close[i] < low_roll_min[i] and close[i] < ema_50_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower channel (reversal signal)
            if close[i] < low_roll_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper channel (reversal signal)
            if close[i] > high_roll_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals