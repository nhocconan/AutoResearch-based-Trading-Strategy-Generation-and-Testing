#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day ATR breakout and volume confirmation.
# Uses ATR-based volatility breakout from previous day's close with volume filter.
# Works in trending markets (bull/bear) by catching breakouts, avoids chop via ADX filter.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "12h_1d_ATRBreakout_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and ADX (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ADX(14) on daily for trend strength
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Previous day's close for breakout calculation
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First period
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # Volume filter: volume > 1.5 * 20-period average on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h[i]) or np.isnan(adx_12h[i]) or np.isnan(prev_close_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when trending (ADX > 25)
        if adx_12h[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: close > previous close + 0.5 * ATR
            if close[i] > prev_close_aligned[i] + 0.5 * atr_12h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: close < previous close - 0.5 * ATR
            elif close[i] < prev_close_aligned[i] - 0.5 * atr_12h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on reverse signal or ATR-based stop
            if close[i] < prev_close_aligned[i] - 0.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on reverse signal or ATR-based stop
            if close[i] > prev_close_aligned[i] + 0.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals