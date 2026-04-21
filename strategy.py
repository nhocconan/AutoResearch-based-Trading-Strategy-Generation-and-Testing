#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_Volume_Regime_v2
Hypothesis: Breakout of Camarilla R1/S1 levels on 12h with 1d volume confirmation and ADX regime filter.
Long when price breaks above R1 with volume > 1.5x 20-period average and ADX < 25 (range/low trend).
Short when price breaks below S1 with volume > 1.5x 20-period average and ADX < 25.
Exit when price reaches opposite level (S1 for long, R1 for short) or reverses at entry level.
Works in both bull/bear by using Camarilla mean-reversion levels and avoiding strong trends via ADX filter.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, R2, S2
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    r2 = prev_close + 1.1 * rang / 6
    s2 = prev_close - 1.1 * rang / 6
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Load 1d data for volume and ADX
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Volume filter: current 1d volume > 1.5 * 20-period average
    if len(volume_1d) >= 20:
        vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        volume_ok_1d = volume_1d > 1.5 * vol_ma
    else:
        volume_ok_1d = np.zeros(len(volume_1d), dtype=bool)
    
    volume_ok_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ok_1d.astype(float))
    
    # ADX(14) on 1d
    if len(high_1d) >= 14:
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr1[0] = high_1d[0] - low_1d[0]  # first bar
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high_1d - np.roll(high_1d, 1)
        down_move = np.roll(low_1d, 1) - low_1d
        up_move[0] = 0
        down_move[0] = 0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR, +DM, -DM
        tr_period = 14
        atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
        adx_ok = adx < 25  # low trend regime
    else:
        adx_ok = np.ones(len(high_1d), dtype=bool)
    
    adx_ok_aligned = align_htf_to_ltf(prices, df_1d, adx_ok.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(volume_ok_1d_aligned[i]) or np.isnan(adx_ok_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long conditions: break above R1 with volume and low ADX
            if (price > r1_aligned[i] and 
                volume_ok_1d_aligned[i] > 0.5 and 
                adx_ok_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 with volume and low ADX
            elif (price < s1_aligned[i] and 
                  volume_ok_1d_aligned[i] > 0.5 and 
                  adx_ok_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reach S1 or reverse below R1
            if price <= s1_aligned[i] or price < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reach R1 or reverse above S1
            if price >= r1_aligned[i] or price > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_Regime_v2"
timeframe = "12h"
leverage = 1.0