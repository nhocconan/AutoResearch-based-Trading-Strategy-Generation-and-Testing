#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA34) and volume confirmation
# Long when price breaks above Donchian upper band AND 1d close > 1d EMA34 AND volume > 1.8 * 20-bar average volume
# Short when price breaks below Donchian lower band AND 1d close < 1d EMA34 AND volume > 1.8 * 20-bar average volume
# Exit when price retests the Donchian midpoint (mean of upper and lower bands)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Donchian(20) provides robust price channel structure
# 1d EMA34 filters for higher timeframe trend alignment (more stable than 12h for 6h)
# Volume confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by following the 1d trend

name = "6h_Donchian20_1dEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian(20) levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels: 20-period high and low
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Align HTF indicators to 6h timeframe (wait for completed 1d bar)
    donchian_20_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_20_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed 1d bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.8 * 20-bar average volume (stricter for fewer trades)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_20_high_aligned[i]) or np.isnan(donchian_20_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper band AND uptrend AND volume confirmation
            if close[i] > donchian_20_high_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower band AND downtrend AND volume confirmation
            elif close[i] < donchian_20_low_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests midpoint from above
            if close[i] <= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests midpoint from below
            if close[i] >= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals