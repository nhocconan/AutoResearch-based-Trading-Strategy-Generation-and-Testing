#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses weekly Camarilla pivot (R4/S4) from 1w OHLC for institutional breakout zones
# Donchian(20) on 6h captures medium-term breakouts with clear structure
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Weekly pivot filter ensures alignment with major trend to avoid counter-trend trades
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes)

name = "6h_Donchian20_1wCamarilla_R4S4_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla levels (R4, S4) for major institutional levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla: R4 = close + 1.5*(high-low)/2, S4 = close - 1.5*(high-low)/2
    camarilla_r4_1w = close_1w + 1.5 * (high_1w - low_1w) / 2
    camarilla_s4_1w = close_1w - 1.5 * (high_1w - low_1w) / 2
    
    # Align to 6h timeframe (wait for weekly close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    # Donchian(20) on 6h for breakout detection
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and volume MA)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high AND above weekly R4 AND volume spike
            if (close[i] > high_ma[i] and 
                close[i] > camarilla_r4_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian low AND below weekly S4 AND volume spike
            elif (close[i] < low_ma[i] and 
                  close[i] < camarilla_s4_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian low (breakdown) OR below weekly S4 (major support)
            if close[i] < low_ma[i] or close[i] < camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high (breakout) OR above weekly R4 (major resistance)
            if close[i] > high_ma[i] or close[i] > camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals