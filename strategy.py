#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly ADX Filter and Volume Confirmation.
Long when price breaks above Donchian upper band with expanding volume and weekly ADX > 25.
Short when price breaks below Donchian lower band with expanding volume and weekly ADX > 25.
Exit when price crosses back to middle line (Donchian mid).
Uses 20-day Donchian channels to capture trends, weekly ADX to filter strong trends,
and volume confirmation to avoid false breakouts.
Designed for low trade frequency (< 25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_adx_volume_v1"
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
    
    # === Donchian Channels (20-period) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === Weekly ADX (14) for trend strength ===
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # True Range
    tr1_w = high_w - low_w
    tr2_w = np.abs(high_w - np.roll(close_w, 1))
    tr3_w = np.abs(low_w - np.roll(close_w, 1))
    tr1_w[0] = 0
    tr2_w[0] = 0
    tr3_w[0] = 0
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    
    # Directional Movement
    dm_plus_w = np.where((high_w - np.roll(high_w, 1)) > (np.roll(low_w, 1) - low_w),
                         np.maximum(high_w - np.roll(high_w, 1), 0), 0)
    dm_minus_w = np.where((np.roll(low_w, 1) - low_w) > (high_w - np.roll(high_w, 1)),
                          np.maximum(np.roll(low_w, 1) - low_w, 0), 0)
    dm_plus_w[0] = 0
    dm_minus_w[0] = 0
    
    # Smoothed values
    tr14_w = pd.Series(tr_w).rolling(window=14, min_periods=14).sum().values
    dm_plus14_w = pd.Series(dm_plus_w).rolling(window=14, min_periods=14).sum().values
    dm_minus14_w = pd.Series(dm_minus_w).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus_w = np.where(tr14_w != 0, 100 * dm_plus14_w / tr14_w, 0)
    di_minus_w = np.where(tr14_w != 0, 100 * dm_minus14_w / tr14_w, 0)
    
    # DX and ADX
    dx_w = np.where((di_plus_w + di_minus_w) != 0,
                    100 * np.abs(di_plus_w - di_minus_w) / (di_plus_w + di_minus_w), 0)
    adx_w = pd.Series(dx_w).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to daily
    adx_w_aligned = align_htf_to_ltf(prices, df_weekly, adx_w)
    
    # === Volume confirmation (daily) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(adx_w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle line
            if close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle line
            if close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average) and strong trend (ADX > 25)
            if vol_ratio[i] < 1.5 or adx_w_aligned[i] < 25:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume and ADX confirmation
            if close[i] > donch_high[i]:
                # Breakout above upper band -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i]:
                # Breakdown below lower band -> short
                position = -1
                signals[i] = -0.25
    
    return signals