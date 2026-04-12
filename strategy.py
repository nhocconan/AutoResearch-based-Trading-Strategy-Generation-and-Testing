#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Reversion_v1
Hypothesis: Mean reversion at weekly/ daily Camarilla levels with 12h trend filter and volume confirmation.
Long when price touches daily L3 in uptrend (12h close > EMA20) with volume spike; short when touches daily H3 in downtrend.
Exit on trend reversal or break of weekly L4/H4. Uses weekly context to filter false breaks in ranging markets.
Designed for 12h timeframe to capture reversals with low frequency and high edge.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_Reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY CONTEXT (regime filter) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (H4, L4 for breakout filter)
    camarilla_h4_1w = np.full(len(close_1w), np.nan)
    camarilla_l4_1w = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i]):
            continue
        range_val = high_1w[i] - low_1w[i]
        camarilla_h4_1w[i] = close_1w[i] + range_val * 1.1 / 4
        camarilla_l4_1w[i] = close_1w[i] - range_val * 1.1 / 4
    
    # Align weekly levels to 12h
    h4_1w_12h = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    l4_1w_12h = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    
    # === DAILY CAMARILLA LEVELS (entry/exit) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H3, L3 for entry, H4/L4 for exit)
    camarilla_h3_1d = np.full(len(close_1d), np.nan)
    camarilla_l3_1d = np.full(len(close_1d), np.nan)
    camarilla_h4_1d = np.full(len(close_1d), np.nan)
    camarilla_l4_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        range_val = high_1d[i] - low_1d[i]
        camarilla_h3_1d[i] = close_1d[i] + range_val * 1.1 / 6
        camarilla_l3_1d[i] = close_1d[i] - range_val * 1.1 / 6
        camarilla_h4_1d[i] = close_1d[i] + range_val * 1.1 / 4
        camarilla_l4_1d[i] = close_1d[i] - range_val * 1.1 / 4
    
    # Align daily levels to 12h
    h3_1d_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    l3_1d_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    h4_1d_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    l4_1d_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # === 12-HOUR TREND FILTER ===
    close_12h = df_1d['close'].values  # Use daily close as proxy for trend (12h aggregates 2x daily)
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_12h = align_htf_to_ltf(prices, df_1d, ema20_12h)  # align daily EMA to 12h
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h3_1d_12h[i]) or np.isnan(l3_1d_12h[i]) or 
            np.isnan(h4_1w_12h[i]) or np.isnan(l4_1w_12h[i]) or
            np.isnan(ema20_12h_12h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h trend using daily EMA
        trend_up = close[i] > ema20_12h_12h[i]
        trend_down = close[i] < ema20_12h_12h[i]
        
        # Long: touch daily L3 in uptrend, volume spike, and above weekly L4 (not in strong weekly downtrend)
        long_signal = (trend_up and 
                      close[i] <= l3_1d_12h[i] * 1.005 and  # Slightly relaxed tolerance for 12h
                      close[i] >= l4_1d_12h[i] * 0.995 and  # Above daily L4
                      vol_ratio[i] > 2.0 and
                      close[i] > l4_1w_12h[i])  # Above weekly L4 to avoid strong weekly downtrend
        
        # Short: touch daily H3 in downtrend, volume spike, and below weekly H4 (not in strong weekly uptrend)
        short_signal = (trend_down and 
                       close[i] >= h3_1d_12h[i] * 0.995 and  # Slightly relaxed tolerance
                       close[i] <= h4_1d_12h[i] * 1.005 and  # Below daily H4
                       vol_ratio[i] > 2.0 and
                       close[i] < h4_1w_12h[i])  # Below weekly H4 to avoid strong weekly uptrend
        
        # Exit: trend reversal or break of daily H4/L4
        exit_long = (position == 1 and 
                    (not trend_up or close[i] >= h4_1d_12h[i]))
        exit_short = (position == -1 and 
                     (not trend_down or close[i] <= l4_1d_12h[i]))
        
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