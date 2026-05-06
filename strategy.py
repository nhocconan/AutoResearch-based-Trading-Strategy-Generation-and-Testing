#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above 1d Donchian upper band AND 1w close > 1w EMA50 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 1d Donchian lower band AND 1w close < 1w EMA50 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 1d Donchian midpoint (mean of upper/lower band)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Donchian channels provide robust price structure that works in both bull and bear markets
# 1w EMA50 filters for higher timeframe trend alignment (proven BTC/ETH edge)
# Volume spike confirmation reduces false breakouts during low participation

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels and 1w EMA50 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper band = highest high over past 20 bars
    # Lower band = lowest low over past 20 bars
    # Midpoint = (upper + lower) / 2
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_midpoint = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1w EMA50 trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe (wait for completed bars)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_midpoint_aligned = align_htf_to_ltf(prices, df_1d, donchian_midpoint)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_midpoint_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper band AND uptrend AND volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower band AND downtrend AND volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests midpoint from above
            if close[i] <= donchian_midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests midpoint from below
            if close[i] >= donchian_midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals