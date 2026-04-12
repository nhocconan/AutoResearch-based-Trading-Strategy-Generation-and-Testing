#!/usr/bin/env python3
"""
4h_1d_AngleBased_Trend_Reversal_v1
Hypothesis: Price reverses at daily Camarilla L3/H3 when 4H price angle (slope of EMA20) is opposite to position, indicating exhaustion.
Long when price touches L3 and 4H EMA20 slope turns up after being down; short when touches H3 and slope turns down after being up.
Exit when price touches opposite H4/L4 or slope reverses. Designed for 4H to work in both bull and bear via mean-reversion at institutional levels.
Target: 50-120 total trades over 4 years (12-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_AngleBased_Trend_Reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H3, L3, H4, L4)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        range_val = high_1d[i] - low_1d[i]
        camarilla_h3[i] = close_1d[i] + range_val * 1.1 / 6
        camarilla_l3[i] = close_1d[i] - range_val * 1.1 / 6
        camarilla_h4[i] = close_1d[i] + range_val * 1.1 / 4
        camarilla_l4[i] = close_1d[i] - range_val * 1.1 / 4
    
    # Align to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 4-HOUR EMA20 AND SLOPE ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Slope: change over 3 periods
    ema20_slope = np.zeros_like(ema20)
    ema20_slope[3:] = (ema20[3:] - ema20[:-3]) / 3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or np.isnan(h4_4h[i]) or 
            np.isnan(l4_4h[i]) or np.isnan(ema20_slope[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price at L3 and EMA20 slope turns up (was <=0, now >0)
        long_signal = (close[i] <= l3_4h[i] * 1.001 and  # Very tight tolerance
                      close[i] >= l4_4h[i] * 0.999 and   # Above L4
                      ema20_slope[i] > 0 and 
                      ema20_slope[i-1] <= 0)
        
        # Short: price at H3 and EMA20 slope turns down (was >=0, now <0)
        short_signal = (close[i] >= h3_4h[i] * 0.999 and   # Very tight tolerance
                       close[i] <= h4_4h[i] * 1.001 and    # Below H4
                       ema20_slope[i] < 0 and 
                       ema20_slope[i-1] >= 0)
        
        # Exit: price touches opposite level or slope reverses
        exit_long = (position == 1 and 
                    (close[i] >= h4_4h[i] or ema20_slope[i] < 0))
        exit_short = (position == -1 and 
                     (close[i] <= l4_4h[i] or ema20_slope[i] > 0))
        
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