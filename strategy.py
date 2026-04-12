#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v4
Hypothesis: On 4h timeframe, take long when price breaks above Camarilla R4 with volume expansion in trending markets (ADX>25), and short when price breaks below Camarilla S4 with volume expansion in trending markets. Uses ADX regime filter to avoid false breakouts in ranging conditions, reducing whipsaws. Designed for 25-40 trades/year by requiring ADX>25 and volume spike (2x average) for entry. Works in bull markets via R4 breakouts and bear markets via S4 breakdowns, while avoiding choppy markets where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation for trend strength (14 period)
    # ADX uses +DI, -DI, and TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data ONCE for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after ADX warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending = adx[i] > 25
        
        # Camarilla breakout conditions
        breakout_r4 = close[i] > camarilla_r4_aligned[i]  # Break above R4
        breakout_s4 = close[i] < camarilla_s4_aligned[i]  # Break below S4
        
        # Volume confirmation: current volume > 2x average (strong breakout)
        volume_spike = volume[i] > vol_ma[i] * 2.0
        
        # Entry conditions: breakout + volume + trending
        long_entry = breakout_r4 and volume_spike and trending
        short_entry = breakout_s4 and volume_spike and trending
        
        # Exit conditions: price returns to Camarilla pivot point (midpoint)
        pivot_point = (high_1d + low_1d + close_1d) / 3
        pivot_point_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
        
        long_exit = close[i] < pivot_point_aligned[i]
        short_exit = close[i] > pivot_point_aligned[i]
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals