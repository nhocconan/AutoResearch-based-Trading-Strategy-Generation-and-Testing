#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation.
# Donchian breakout captures momentum in both bull and bear markets.
# 1d EMA50 ensures we only trade in the direction of the daily trend.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Target: 25-50 trades/year (100-200 total over 4 years) to minimize fee drag.
name = "4h_Donchian20_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA50 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper channel AND uptrend AND volume spike
            if close[i] > upper_channel[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND downtrend AND volume spike
            elif close[i] < lower_channel[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower channel OR trend reverses
            if close[i] < lower_channel[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper channel OR trend reverses
            if close[i] > upper_channel[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals