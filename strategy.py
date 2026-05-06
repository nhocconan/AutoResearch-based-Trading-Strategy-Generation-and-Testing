#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation
# Long when price > Alligator Jaw (teeth) AND 1d close > 1d EMA34 AND volume > 2.0 * 20-bar average volume
# Short when price < Alligator Jaw (teeth) AND 1d close < 1d EMA34 AND volume > 2.0 * 20-bar average volume
# Exit when price crosses back below/above Alligator Lips
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams Alligator (SMAs: 13,8,5) provides trend identification with built-in smoothing
# 1d EMA34 filters for higher timeframe trend alignment (proven BTC/ETH edge from Camarilla winners)
# Volume spike confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by following the 1d trend

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate 6h Williams Alligator and 1d EMA34 ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_6h) < 13 or len(df_1d) < 34:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (SMAs: Jaw=13, Teeth=8, Lips=5)
    close_6h_series = pd.Series(close_6h)
    alligator_jaw = close_6h_series.rolling(window=13, min_periods=13).mean().values  # Blue line
    alligator_teeth = close_6h_series.rolling(window=8, min_periods=8).mean().values    # Red line
    alligator_lips = close_6h_series.rolling(window=5, min_periods=5).mean().values   # Green line
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed bars)
    alligator_jaw_aligned = align_htf_to_ltf(prices, df_6h, alligator_jaw)
    alligator_teeth_aligned = align_htf_to_ltf(prices, df_6h, alligator_teeth)
    alligator_lips_aligned = align_htf_to_ltf(prices, df_6h, alligator_lips)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(alligator_jaw_aligned[i]) or np.isnan(alligator_teeth_aligned[i]) or 
            np.isnan(alligator_lips_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Alligator Jaw (teeth) AND uptrend AND volume spike
            if close[i] > alligator_jaw_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < Alligator Jaw (teeth) AND downtrend AND volume spike
            elif close[i] < alligator_jaw_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Alligator Lips
            if close[i] < alligator_lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Alligator Lips
            if close[i] > alligator_lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals