#!/usr/bin/env python3
"""
1d Donchian Breakout with 1-week ADX Trend and Volume Confirmation.
Long when price breaks above Donchian upper band with strong weekly trend and volume.
Short when price breaks below Donchian lower band with strong weekly trend and volume.
Exit when price crosses opposite Donchian band or trend weakens.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_adx_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W ADX TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate ADX on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx) | np.isinf(dx), 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx = np.where(np.isnan(adx), 0, adx)
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # === DAILY DONCHIAN CHANNELS ===
    donch_len = 20
    upper = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lower = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donch_len, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Strong trend filter
        strong_trend = adx_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price crosses below lower band OR trend weakens
            if close[i] < lower[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper band OR trend weakens
            if close[i] > upper[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation and strong trend
            if volume[i] <= vol_ma[i] or not strong_trend:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with trend alignment
            if close[i] > upper[i]:
                # Breakout above upper band -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lower[i]:
                # Breakdown below lower band -> short
                position = -1
                signals[i] = -0.25
    
    return signals