#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_Breakout_v1
Hypothesis: Breakout at weekly and daily Camarilla H4/L4 levels with trend continuation.
Long when price breaks above H4 (weekly or daily) with momentum and volume.
Short when price breaks below L4 (weekly or daily) with momentum and volume.
Uses 6h timeframe to capture multi-day moves in both bull and bear markets.
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Camarilla_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY CAMARILLA LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (H4, L4)
    camarilla_h4_1w = np.full(len(close_1w), np.nan)
    camarilla_l4_1w = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i]):
            continue
        range_val = high_1w[i] - low_1w[i]
        camarilla_h4_1w[i] = close_1w[i] + range_val * 1.1 / 4
        camarilla_l4_1w[i] = close_1w[i] - range_val * 1.1 / 4
    
    # Align to 6h timeframe
    h4_1w_6h = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    l4_1w_6h = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H4, L4)
    camarilla_h4_1d = np.full(len(close_1d), np.nan)
    camarilla_l4_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        range_val = high_1d[i] - low_1d[i]
        camarilla_h4_1d[i] = close_1d[i] + range_val * 1.1 / 4
        camarilla_l4_1d[i] = close_1d[i] - range_val * 1.1 / 4
    
    # Align to 6h timeframe
    h4_1d_6h = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    l4_1d_6h = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # === MOMENTUM FILTER (6h ROC) ===
    roc_period = 12  # 3 days
    roc = np.full(n, np.nan)
    if n >= roc_period:
        roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period] * 100
    
    # === VOLUME FILTER ===
    vol_ma_period = 24  # 4 days
    vol_ma = np.full(n, np.nan)
    if n >= vol_ma_period:
        vol_ma_series = pd.Series(volume)
        vol_ma[vol_ma_period:] = vol_ma_series.rolling(window=vol_ma_period, min_periods=vol_ma_period).mean().values[vol_ma_period:]
    
    vol_ratio = np.full(n, np.nan)
    valid_vol = (vol_ma > 0) & ~np.isnan(vol_ma)
    vol_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, roc_period, vol_ma_period)
    
    for i in range(start_idx, n):
        # Skip if not ready
        if (np.isnan(h4_1w_6h[i]) or np.isnan(l4_1w_6h[i]) or 
            np.isnan(h4_1d_6h[i]) or np.isnan(l4_1d_6h[i]) or
            np.isnan(roc[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        # Long: price breaks above H4 (weekly OR daily) with upward momentum and volume
        weekly_h4_break = close[i] > h4_1w_6h[i]
        daily_h4_break = close[i] > h4_1d_6h[i]
        h4_break = weekly_h4_break or daily_h4_break
        
        # Short: price breaks below L4 (weekly OR daily) with downward momentum and volume
        weekly_l4_break = close[i] < l4_1w_6h[i]
        daily_l4_break = close[i] < l4_1d_6h[i]
        l4_break = weekly_l4_break or daily_l4_break
        
        # Momentum and volume filters
        mom_filter = roc[i] > 0  # upward momentum for long, downward for short
        vol_filter = vol_ratio[i] > 1.5
        
        # Long entry
        long_signal = h4_break and mom_filter and vol_filter
        
        # Short entry
        short_signal = (not mom_filter) and l4_break and vol_filter  # roc[i] < 0 for short
        
        # Exit on opposite breakout or momentum reversal
        exit_long = position == 1 and (
            (close[i] < l4_1w_6h[i] and close[i] < l4_1d_6h[i]) or  # break below both L4s
            roc[i] < -0.5  # strong downward momentum
        )
        
        exit_short = position == -1 and (
            (close[i] > h4_1w_6h[i] and close[i] > h4_1d_6h[i]) or  # break above both H4s
            roc[i] > 0.5  # strong upward momentum
        )
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals