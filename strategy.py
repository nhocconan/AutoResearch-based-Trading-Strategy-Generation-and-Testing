#!/usr/bin/env python3
"""
Hypothesis: 1-day price closes above/below weekly Donchian channel (20-period) with volume confirmation.
In strong uptrends: price closes above weekly Donchian upper band -> long.
In strong downtrends: price closes below weekly Donchian lower band -> short.
Weekly Donchian provides robust trend structure, daily close ensures follow-through,
volume filters weak breakouts. Target: 10-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian upper and lower bands"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=np.float64), np.full_like(high, np.nan, dtype=np.float64)
    
    upper = np.full_like(high, np.nan, dtype=np.float64)
    lower = np.full_like(high, np.nan, dtype=np.float64)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_donch_upper, wk_donch_lower = calculate_donchian_channels(wk_high, wk_low, 20)
    wk_donch_upper_aligned = align_htf_to_ltf(prices, df_1w, wk_donch_upper)
    wk_donch_lower_aligned = align_htf_to_ltf(prices, df_1w, wk_donch_lower)
    
    # Get daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(wk_donch_upper_aligned[i]) or np.isnan(wk_donch_lower_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current daily close and volume
        price_close = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current weekly Donchian levels
        donch_upper = wk_donch_upper_aligned[i]
        donch_lower = wk_donch_lower_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        if position == 0:
            # Long: daily close above weekly Donchian upper + volume
            if price_close > donch_upper and vol_filter:
                signals[i] = size
                position = 1
            # Short: daily close below weekly Donchian lower + volume
            elif price_close < donch_lower and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: daily close back below weekly Donchian upper
            if price_close < donch_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: daily close back above weekly Donchian lower
            if price_close > donch_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0