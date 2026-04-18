#!/usr/bin/env python3
"""
12h_KAMA_With_Volume_And_Chop_Filter
Hypothesis: KAMA adapts to market noise, providing a smoother trend signal than traditional MAs.
In trending markets, price stays above/below KAMA; in ranging markets, price oscillates around it.
Combined with volume confirmation (to avoid false breakouts) and Choppiness Index (to avoid choppy markets),
this strategy aims to capture strong trends while avoiding whipsaws. Works in both bull and bear markets
by only trading when trend is strong and volume confirms. Targets 12-37 trades/year with position size 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for Choppiness Index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR (14-period) for Choppiness Index
    atr_1d = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    # Calculate highest high and lowest low over 14 periods
    hh_1d = np.full(len(high_1d), np.nan)
    ll_1d = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        hh_1d[i] = np.max(high_1d[i-13:i+1])
        ll_1d[i] = np.min(low_1d[i-13:i+1])
    
    # Calculate Choppiness Index (14-period)
    chop_1d = np.full(len(high_1d), 50.0)  # default to middle
    for i in range(14, len(high_1d)):
        if hh_1d[i] > ll_1d[i] and atr_1d[i] > 0:
            sum_atr = np.sum(tr_1d[i-13:i+1])
            chop_1d[i] = 100 * np.log10(sum_atr / (hh_1d[i] - ll_1d[i])) / np.log10(14)
        else:
            chop_1d[i] = 50.0
    
    # Align Choppiness Index to 12h timeframe (wait for daily bar close)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate KAMA (10-period) on close
    # Efficiency Ratio (ER) = |net change| / sum of absolute changes
    change = np.abs(np.diff(close, n=1))
    abs_change = np.abs(np.diff(close, n=1))
    
    # Pad arrays for calculation
    change_pad = np.concatenate([[np.nan], change])
    abs_change_pad = np.concatenate([[np.nan], abs_change])
    
    er = np.full(n, np.nan)
    for i in range(10, n):
        if np.sum(abs_change_pad[i-9:i+1]) > 0:
            er[i] = np.nansum(change_pad[i-9:i+1]) / np.nansum(abs_change_pad[i-9:i+1])
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = np.full(n, np.nan)
    for i in range(10, n):
        sc[i] = (er[i] * (0.6667 - 0.0645) + 0.0645) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # need volume MA and KAMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_12h[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Chop filter: only trade when market is not too choppy (CHOP < 61.8 = trending)
        chop_filter = chop_12h[i] < 61.8
        
        if position == 0:
            # Long entry: price crosses above KAMA with volume confirmation and trend filter
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and vol_confirmed and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below KAMA with volume confirmation and trend filter
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and vol_confirmed and chop_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_With_Volume_And_Chop_Filter"
timeframe = "12h"
leverage = 1.0