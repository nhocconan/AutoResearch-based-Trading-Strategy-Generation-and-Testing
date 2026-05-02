#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses 4h primary timeframe for optimal trade frequency (target: 75-200 trades over 4 years)
# Donchian channels provide clear structure-based breakout levels
# 12h EMA50 trend filter ensures alignment with higher timeframe momentum
# Volume spike (2.0x 20-period average) confirms institutional participation
# Designed with tight entry conditions to minimize fee drag while maintaining edge in both bull and bear markets
# Target: 100-180 total trades over 4 years (25-45/year) - within proven winning range

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for HTF calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian channels (20-period) from 4h data
    # Upper band = 20-period high, Lower band = 20-period low
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, volume MA and HTF data alignment)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper band + price > 12h EMA50 + volume spike
            if close[i] > high_ma[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band + price < 12h EMA50 + volume spike
            elif close[i] < low_ma[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower band (reversal signal)
            if close[i] < low_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper band (reversal signal)
            if close[i] > high_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals