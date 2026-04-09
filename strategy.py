#!/usr/bin/env python3
# 6h_adx_di_volume_v1
# Hypothesis: 6h strategy using 12h ADX/DI for regime and trend direction,
# with volume confirmation for entry timing. Designed for low trade frequency
# (target: 12-37 trades/year) to avoid fee drag. Works in bull/bear by using
# ADX > 25 for trending regime and DI crossover for direction. Uses discrete
# sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_di_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for ADX/DI regime and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) and DI
    # True Range
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_12h) - pd.Series(high_12h).shift(1)
    down_move = pd.Series(low_12h).shift(1) - pd.Series(low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr_12h + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_12h + 1e-10)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to LTF
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_12h, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_12h, minus_di)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or
            np.isnan(minus_di_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime and direction from 12h ADX/DI
        trending = adx_aligned[i] > 25
        DI_cross_up = plus_di_aligned[i] > minus_di_aligned[i]
        DI_cross_down = plus_di_aligned[i] < minus_di_aligned[i]
        
        if position == 1:  # Long position
            # Exit: trend ends OR DI crosses down
            if not trending or not DI_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend ends OR DI crosses up
            if not trending or not DI_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed and trending:
                # Long conditions: DI cross up
                if DI_cross_up:
                    position = 1
                    signals[i] = 0.25
                # Short conditions: DI cross down
                elif DI_cross_down:
                    position = -1
                    signals[i] = -0.25
    
    return signals