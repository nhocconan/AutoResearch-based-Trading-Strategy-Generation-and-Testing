#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian channel breakout with weekly trend filter and volume confirmation.
Trades breakouts above 1-day Donchian high (or below low) when weekly trend aligns and volume exceeds 1.5x weekly average.
Designed to capture strong trends in both bull and bear markets by using weekly trend filter and volume confirmation.
Target: 10-20 trades/year per symbol (40-80 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 1-day Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian high and low (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1-day timeframe (then to lower timeframe)
    donchian_high_aligned_1d = align_htf_to_ltf(df_1d, df_1d, donchian_high)  # identity for same TF
    donchian_low_aligned_1d = align_htf_to_ltf(df_1d, df_1d, donchian_low)
    
    # Now align to the primary timeframe (1d -> 1d is identity, but we'll keep for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_aligned_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_aligned_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA(10) for trend
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Get weekly data for volume filter
    volume_1w = df_1w['volume'].values
    vol_ma_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, weekly EMA, and weekly volume MA
    start_idx = max(20, 10, 10)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_10_1w_aligned[i]) or np.isnan(vol_ma_10_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current Donchian levels
        dch_high = donchian_high_aligned[i]
        dch_low = donchian_low_aligned[i]
        
        # Weekly trend and volume
        trend_1w = ema_10_1w_aligned[i]
        vol_ma_1w = vol_ma_10_1w_aligned[i]
        
        # Volume filter: volume > 1.5x weekly average
        vol_filter = vol_now > 1.5 * vol_ma_1w
        
        # Entry conditions: Donchian breakout with volume and weekly trend alignment
        if position == 0:
            # Long: price breaks above Donchian high with volume + weekly uptrend
            if price_now > dch_high and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume + weekly downtrend
            elif price_now < dch_low and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below Donchian low or weekly trend turns down
            if price_now < dch_low or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above Donchian high or weekly trend turns up
            if price_now > dch_high or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0