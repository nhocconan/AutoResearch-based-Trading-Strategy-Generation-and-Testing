#!/usr/bin/env python3
"""
1h_4D_Donchian_UpperLower_With_Volume_Spike
Hypothesis: 1-hour Donchian breakouts above the 4-day upper band (or below the 4-day lower band) with volume confirmation and 1d trend filter capture institutional breakout moves. Using 4d bands (vs 20-period) reduces noise and increases breakout quality. Volume spike (>1.5x 24-period average) confirms participation, while 1d EMA50 trend filter ensures directional alignment. Target: 20-40 trades/year per symbol.
"""

name = "1h_4D_Donchian_UpperLower_With_Volume_Spike"
timeframe = "1h"
leverage = 1.0

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
    
    # Volume spike: >1.5x 24-period average (1 day of 1h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 4-hour data for 4-day Donchian channels (96 periods of 4h = 4 days)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 96:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4-day Donchian channels: highest high and lowest low of last 96 periods
    dh_4h = pd.Series(high_4h).rolling(window=96, min_periods=96).max().values
    dl_4h = pd.Series(low_4h).rolling(window=96, min_periods=96).min().values
    
    # 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1h timeframe
    dh_4h_aligned = align_htf_to_ltf(prices, df_4h, dh_4h)
    dl_4h_aligned = align_htf_to_ltf(prices, df_4h, dl_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(dh_4h_aligned[i]) or
            np.isnan(dl_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 4d Donchian high + 1d EMA50 uptrend + volume spike
            if (close[i] > dh_4h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below 4d Donchian low + 1d EMA50 downtrend + volume spike
            elif (close[i] < dl_4h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 4d Donchian low OR closes below 1d EMA50
            if (close[i] < dl_4h_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 4d Donchian high OR closes above 1d EMA50
            if (close[i] > dh_4h_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals