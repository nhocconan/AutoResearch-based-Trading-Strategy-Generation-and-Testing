#!/usr/bin/env python3
"""
12h Trading Range Breakout with Volume and Trend Confirmation
Hypothesis: In trending markets, price breaks above/below prior 12h high/low with volume
indicate institutional participation. Works in both bull and bear by following breakout
direction. Uses 1w trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend
    close_1w = df_1w['close'].values
    ema_34_1w = np.zeros_like(close_1w)
    ema_34_1w[:] = np.nan
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2 + ema_34_1w[i-1] * 33) / 35
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12-period high/low for breakout levels (using prior 12h bar)
    highest_12 = np.full(n, np.nan)
    lowest_12 = np.full(n, np.nan)
    for i in range(12, n):
        highest_12[i] = np.max(high[i-12:i])
        lowest_12[i] = np.min(low[i-12:i])
    
    # Volume spike: current volume > 2x average of last 12 periods
    vol_avg = np.full(n, np.nan)
    for i in range(12, n):
        vol_avg[i] = np.mean(volume[i-12:i])
    vol_spike = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure we have enough data for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(highest_12[i]) or np.isnan(lowest_12[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_1d_aligned[i] > 0.01 * close[i]  # At least 1% ATR
        
        if position == 0:
            # Long: break above 12h high with volume in uptrend
            if (close[i] > highest_12[i] and 
                vol_spike[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: break below 12h low with volume in downtrend
            elif (close[i] < lowest_12[i] and 
                  vol_spike[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: break below 12h low or trend reversal
            if (close[i] < lowest_12[i] or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: break above 12h high or trend reversal
            if (close[i] > highest_12[i] or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Trend_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0