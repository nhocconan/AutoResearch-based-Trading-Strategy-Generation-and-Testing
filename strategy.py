#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND 1d close > 1d EMA34 AND volume > 1.5 * 20-bar average volume
# Short when price breaks below 4h Donchian lower band AND 1d close < 1d EMA34 AND volume > 1.5 * 20-bar average volume
# Exit when price crosses the 4h Donchian middle band (20-period average)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-150 total trades over 4 years (19-37/year) for 4h timeframe
# Donchian channels provide clear trend-following structure with defined breakout levels
# 1d EMA34 filters for higher timeframe trend alignment to avoid counter-trend trades
# Volume confirmation reduces false breakouts during low participation periods
# Works in both bull and bear markets by following the 1d trend direction

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v2"
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
    
    # Calculate 4h Donchian channels and 1d EMA34 ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 34:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0  # Middle band
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed bars)
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_4h, mid_20)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper band AND uptrend AND volume spike
            if close[i] > high_20_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower band AND downtrend AND volume spike
            elif close[i] < low_20_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle band
            if close[i] < mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle band
            if close[i] > mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals