#!/usr/bin/env python3
"""
6h_12h_1d_Camarilla_Pullback_With_Flow
Hypothesis: Pullback to Camarilla pivot levels (H3/L3) on 1-day data with 12h trend filter and volume surge confirmation.
Long when price pulls back to L3 in uptrend (12h close > 12h EMA20) with volume spike; short when pulls back to H3 in downtrend.
Uses Camarilla levels from daily timeframe for institutional-grade support/resistance.
Designed for 6h timeframe to capture mean-reversion bounces in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Camarilla_Pullback_With_Flow"
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
    
    # === 1-DAY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            camarilla_h3[i] = camarilla_l3[i] = camarilla_h4[i] = camarilla_l4[i] = np.nan
            continue
            
        range_val = high_1d[i] - low_1d[i]
        camarilla_h3[i] = close_1d[i] + range_val * 1.1 / 6
        camarilla_l3[i] = close_1d[i] - range_val * 1.1 / 6
        camarilla_h4[i] = close_1d[i] + range_val * 1.1 / 4
        camarilla_l4[i] = close_1d[i] - range_val * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    h3_6h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_6h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_6h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_6h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 12-HOUR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA20 of 12-hour close
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_6h = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h3_6h[i]) or np.isnan(l3_6h[i]) or np.isnan(ema20_12h_6h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h trend: up if close > EMA20, down if close < EMA20
        # Get current 12h close aligned to 6h
        if len(df_12h) >= 2:
            close_12h_arr = df_12h['close'].values
            close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h_arr)
            trend_up = close_12h_aligned[i] > ema20_12h_6h[i]
            trend_down = close_12h_aligned[i] < ema20_12h_6h[i]
        else:
            trend_up = trend_down = False
        
        # Long: pullback to L3 in uptrend with volume surge
        long_signal = (trend_up and 
                      close[i] <= l3_6h[i] * 1.005 and  # Allow small tolerance
                      close[i] >= l4_6h[i] * 0.995 and  # But above L4 to avoid breakdown
                      vol_ratio[i] > 1.8)
        
        # Short: pullback to H3 in downtrend with volume surge
        short_signal = (trend_down and 
                       close[i] >= h3_6h[i] * 0.995 and  # Allow small tolerance
                       close[i] <= h4_6h[i] * 1.005 and  # But below H4 to avoid breakout
                       vol_ratio[i] > 1.8)
        
        # Exit conditions: reversal of trend or price moves past H4/L4
        exit_long = (position == 1 and 
                    (not trend_up or close[i] >= h4_6h[i]))
        exit_short = (position == -1 and 
                     (not trend_down or close[i] <= l4_6h[i]))
        
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