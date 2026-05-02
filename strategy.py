#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakout captures strong momentum moves in both bull and bear markets
# 1d EMA50 provides higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>1.5 x 20-period EMA) confirms breakout validity
# Discrete position sizing (0.25) minimizes fee churn and controls drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and EMA calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper band with volume confirmation and uptrend
            if close[i] > highest_20[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with volume confirmation and downtrend
            elif close[i] < lowest_20[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price closes below Donchian middle (10-period) OR trend changes to downtrend
            # Using 10-period midpoint as exit signal for faster re-entry
            midpoint_10 = (pd.Series(high).rolling(window=10, min_periods=10).max().values[i] + 
                          pd.Series(low).rolling(window=10, min_periods=10).min().values[i]) / 2
            if close[i] < midpoint_10 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price closes above Donchian middle (10-period) OR trend changes to uptrend
            midpoint_10 = (pd.Series(high).rolling(window=10, min_periods=10).max().values[i] + 
                          pd.Series(low).rolling(window=10, min_periods=10).min().values[i]) / 2
            if close[i] > midpoint_10 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals