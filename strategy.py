#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly EMA(34) for trend direction
    ema_34_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(df_1w)):
            ema_34_1w[i] = (close_1w[i] * 2 + ema_34_1w[i-1] * 33) / 34
    
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily volume average (20-period)
    vol_avg_20 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        vol_avg_20[19] = np.mean(volume_1d[:20])
        for i in range(20, len(df_1d)):
            vol_avg_20[i] = (volume_1d[i] + vol_avg_20[i-1] * 19) / 20
    
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # 12-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        vol_filter = volume[i] > (vol_avg_20_aligned[i] * 1.5)
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high AND weekly uptrend AND volume spike
            if close[i] > donch_high[i] and close[i] > ema_34_1w_aligned[i] and vol_filter:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 12h Donchian low AND weekly downtrend AND volume spike
            elif close[i] < donch_low[i] and close[i] < ema_34_1w_aligned[i] and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 12h Donchian low
            if close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 12h Donchian high
            if close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_EMA34_Trend_Volume_Donchian_Breakout"
timeframe = "12h"
leverage = 1.0