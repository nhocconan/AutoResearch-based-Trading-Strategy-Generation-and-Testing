#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_1dTrend
Hypothesis: Camarilla pivot breakout on 12h timeframe with 1-day trend filter and volume confirmation.
In uptrend (price > 1-day EMA34), long at R3 breakout with volume.
In downtrend (price < 1-day EMA34), short at S3 breakdown with volume.
Uses Camarilla levels for institutional-grade support/resistance, reducing false breakouts.
Target: 20-30 trades per year (~80-120 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_Breakout_1dTrend"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close
    c = close
    h = high
    l = low
    r3 = c + (range_val * 1.1 / 2)
    r2 = c + (range_val * 1.1 / 4)
    r1 = c + (range_val * 1.1 / 6)
    s1 = c - (range_val * 1.1 / 6)
    s2 = c - (range_val * 1.1 / 4)
    s3 = c - (range_val * 1.1 / 2)
    return r3, r2, r1, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for each 12h bar using prior bar's HLC
    r3 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        r3[i], r2[i], r1[i], s1[i], s2[i], s3[i] = calculate_camarilla(high[i-1], low[i-1], close[i-1])
    
    # Volume ratio: current volume / 30-period average volume
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need 35 periods for EMA34 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA34
        uptrend_regime = close[i] > ema_34_1d_aligned[i]
        downtrend_regime = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: volume > 1.8x average
        volume_confirm = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: close breaks above R3 in uptrend regime + volume
            long_entry = (close[i] > r3[i]) and uptrend_regime and volume_confirm
            # Short: close breaks below S3 in downtrend regime + volume
            short_entry = (close[i] < s3[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below R1 or regime changes to downtrend
            if (close[i] < r1[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above S1 or regime changes to uptrend
            if (close[i] > s1[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals