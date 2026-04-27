#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian(20) breakout with 1-week EMA trend filter and volume confirmation.
Trades breakouts above Donchian upper band or below lower band when weekly trend confirms direction
and volume exceeds 1-week average. Uses weekly trend to filter direction in both bull and bear markets.
Target: 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1-day timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1-week data for trend filter and volume average
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1-week volume average (20-period)
    volume_1w = df_1w['volume'].values
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian levels, weekly EMA, and weekly volume average
    start_idx = max(20, 20, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current Donchian levels
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        # Weekly trend and volume filters
        trend_1w = ema_20_1w_aligned[i]
        vol_avg_1w = vol_avg_20_1w_aligned[i]
        
        # Volume filter: volume > 1.5x 1-week average
        vol_filter = vol_now > 1.5 * vol_avg_1w
        
        # Entry conditions: Donchian breakout with weekly trend and volume confirmation
        if position == 0:
            # Long: breakout above upper band with weekly uptrend + volume
            if price_now > upper and trend_1w > 0 and vol_filter:
                signals[i] = size
                position = 1
            # Short: breakout below lower band with weekly downtrend + volume
            elif price_now < lower and trend_1w < 0 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below weekly EMA or Donchian lower band
            if price_now < trend_1w or price_now < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above weekly EMA or Donchian upper band
            if price_now > trend_1w or price_now > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wEMATrend_Volume"
timeframe = "1d"
leverage = 1.0