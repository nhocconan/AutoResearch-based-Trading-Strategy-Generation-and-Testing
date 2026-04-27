#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
Works in both bull and bear markets by only taking breakouts in the direction of the weekly trend.
Uses discrete position sizing (0.25) and exits on opposite Donchian breakout or trend reversal.
Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.
"""

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
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily Donchian channels (20-period) from previous completed daily bar
    df_1d = get_htf_data(prices, '1d')
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (already aligned, but using helper for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need 20-period Donchian, 50-period EMA, 20-period volume average
    start_idx = max(20, 50, 20) + 1  # +1 for shift(1) in Donchian
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with weekly trend alignment and volume spike
            long_condition = (close_val > upper and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < lower and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price breaks below Donchian low OR weekly trend reverses (close < weekly EMA50)
            if close_val < lower or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR weekly trend reverses (close > weekly EMA50)
            if close_val > upper or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0