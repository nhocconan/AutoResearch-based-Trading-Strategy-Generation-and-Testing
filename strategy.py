#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_VolumeSpike
Hypothesis: Donchian(20) breakouts on 6h with 12h EMA50 trend filter and volume confirmation. 
Targets 50-150 total trades over 4 years by requiring confluence of 12h trend, volume spike, and price breaking Donchian channels. 
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via 12h trend filter. 
Primary timeframe: 6h, HTF: 12h for trend and Donchian calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter and Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper channel: highest high over last 20 12h periods
    high_series = pd.Series(high_12h)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over last 20 12h periods
    low_series = pd.Series(low_12h)
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h (no extra delay needed as they're based on completed 12h candles)
    upper_channel_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    # Volume spike: volume > 2.0x 20-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 12h EMA, 20 for Donchian, 20 for volume median
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above upper Donchian channel with volume spike and uptrend (close > EMA50_12h)
            long_entry = (close_val > upper_channel_aligned[i]) and vol_spike and (close_val > ema_50_val)
            # Short: price breaks below lower Donchian channel with volume spike and downtrend (close < EMA50_12h)
            short_entry = (close_val < lower_channel_aligned[i]) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or price re-enters Donchian channel
            if close_val < ema_50_val or close_val < lower_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or price re-enters Donchian channel
            if close_val > ema_50_val or close_val > upper_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0