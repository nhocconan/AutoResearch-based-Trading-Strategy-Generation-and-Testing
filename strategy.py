#!/usr/bin/env python3
"""
6h_WeeklyPivot_Momentum
Hypothesis: Use weekly pivot points (from 1w) to establish major support/resistance zones,
combined with 1d EMA trend and volume confirmation on 6basis. Long when price breaks above
weekly R1 with 1d uptrend and volume surge; short when breaks below weekly S1 with 1d downtrend.
Uses 6basis ATR to filter low volatility chop. Designed for 10-20 trades/year by requiring
confluence of weekly structure, daily trend, and volume.
"""

name = "6h_WeeklyPivot_Momentum"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's H/L/C)
    # We'll use the most recent completed week's data
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # Pivot = (H + L + C)/3
    wp = (wh + wl + wc) / 3.0
    # R1 = 2*P - L
    wr1 = 2 * wp - wl
    # S1 = 2*P - H
    ws1 = 2 * wp - wh
    
    # Align weekly pivot levels to 6basis (these update only when new weekly bar completes)
    wp_aligned = align_htf_to_ltf(prices, df_1w, wp)
    wr1_aligned = align_htf_to_ltf(prices, df_1w, wr1)
    ws1_aligned = align_htf_to_ltf(prices, df_1w, ws1)
    
    # Get daily data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d average volume (20-period) for volume filter
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 6basis ATR for volatility filter (6-period ATR, 24-period average for reference)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6 = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    atr_avg_6 = pd.Series(atr_6).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (1), daily EMA (34), daily vol (20), 6basis ATR avg (24)
    start_idx = max(1, 34, 20, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(wp_aligned[i]) or np.isnan(wr1_aligned[i]) or np.isnan(ws1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or np.isnan(atr_avg_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current 6basis volume > 2x average 1d volume (scaled to 6basis)
        vol_6h = volume[i]
        vol_6h_equiv = vol_avg_1d_aligned[i] / 4.0  # 1d = 4x 6basis
        volume_filter = vol_6h > vol_6h_equiv * 2.0
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_6[i] > atr_avg_6[i] * 0.5
        
        if position == 0:
            # Long: price breaks above weekly R1 with uptrend and volume
            if close[i] > wr1_aligned[i] and uptrend_1d and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with downtrend and volume
            elif close[i] < ws1_aligned[i] and downtrend_1d and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to weekly pivot or trend breaks
            if close[i] < wp_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to weekly pivot or trend breaks
            if close[i] > wp_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals