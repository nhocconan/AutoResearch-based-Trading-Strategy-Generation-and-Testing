#!/usr/bin/env python3

"""
Hypothesis: 1-hour Donchian Channel Breakout with 4-hour ADX trend filter and 1-day volume confirmation.
Trades breakouts of 20-period Donchian channels only when 4-hour ADX > 25 (trending market) and volume confirms.
Uses 1-hour for entry timing to capture momentum, while 4-hour trend and 1-day volume filter out false breakouts.
Designed for low trade frequency (15-35 trades/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    plus_dm = np.diff(high)
    minus_dm = np.diff(low)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = np.maximum(np.abs(np.diff(high)), np.maximum(np.abs(np.diff(low)), np.abs(np.diff(close))))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for ADX trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h ADX for trend filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Load 1d data for volume confirmation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-day average volume
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 1-hour Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or 
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend and volume filters
        strong_trend = adx_4h_aligned[i] > 25
        vol_confirm = volume[i] > 1.5 * avg_vol_1d_aligned[i]
        
        if position == 0 and strong_trend and vol_confirm:
            # Long breakout
            if close[i] > high_max_20[i]:
                signals[i] = 0.20
                position = 1
            # Short breakout
            elif close[i] < low_min_20[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: opposite Donchian breakout or trend weakening
            exit_signal = False
            
            if position == 1:
                # Exit long on lower band break or weak trend
                if close[i] < low_min_20[i] or adx_4h_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short on upper band break or weak trend
                if close[i] > high_max_20[i] or adx_4h_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian_Breakout_4hADX_1dVolume"
timeframe = "1h"
leverage = 1.0