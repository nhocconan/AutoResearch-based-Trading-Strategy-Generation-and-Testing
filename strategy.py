#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d volume spike and ADX regime filter.
- Primary timeframe: 4h for execution, HTF: 1d for volume confirmation and ADX.
- Camarilla levels calculated from previous 1d OHLC: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4.
- Volume confirmation: current 4h volume > 2.0 * 20-period volume MA (on 4h).
- ADX > 25 indicates trending market (breakout continuation), ADX < 20 indicates ranging (fade breakouts).
- Entry: Long when close > H3 and volume spike and ADX > 25.
         Short when close < L3 and volume spike and ADX > 25.
         In ranging (ADX < 20): Long when close < L3 and volume spike (mean reversion at support),
                                Short when close > H3 and volume spike (mean reversion at resistance).
- Exit: Opposite Camarilla level touch (close crosses back below H3 for long, above L3 for short) or ADX regime shift.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d True Range for ADX
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement for ADX
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = df_1d['close'].values + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 4
    camarilla_l3 = df_1d['close'].values - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 4
    
    # Align 1d indicators to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        h3_val = camarilla_h3_aligned[i]
        l3_val = camarilla_l3_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if adx_val > 25:  # Trending regime: breakout continuation
                    # Breakout entries: follow the break
                    if close_val > h3_val:
                        signals[i] = 0.25
                        position = 1
                    elif close_val < l3_val:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime (ADX < 20): fade breakouts
                    # Mean reversion: fade extreme moves
                    if close_val < l3_val:
                        signals[i] = 0.25
                        position = 1
                    elif close_val > h3_val:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price crosses back below H3 or ADX drops to ranging
            if close_val < h3_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above L3 or ADX drops to ranging
            if close_val > l3_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dADX_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0