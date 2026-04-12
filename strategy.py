#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_Breakout_Trend
Hypothesis: Use weekly (W1) market regime to filter daily (D1) Camarilla H3/L3 breakouts on 4h timeframe.
In bull regime (price above weekly EMA40), only take longs at D1 H3 breakouts with volume confirmation.
In bear regime (price below weekly EMA40), only take shorts at D1 L3 breakdowns with volume confirmation.
In range regime (price near weekly EMA40), stand flat. This avoids counter-trend trades and reduces whipsaws.
Target: 50-120 total trades over 4 years (12-30/year) on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_Breakout_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY REGIME FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_4h = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or np.isnan(ema40_1w_4h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_above_weekly_ema = close[i] > ema40_1w_4h[i]
        price_below_weekly_ema = close[i] < ema40_1w_4h[i]
        
        # Long: price above weekly EMA40 (bull regime) + break above daily H3 + volume surge
        long_signal = (price_above_weekly_ema and 
                      close[i] > h3_4h[i] * 1.001 and 
                      vol_ratio[i] > 1.8)
        
        # Short: price below weekly EMA40 (bear regime) + break below daily L3 + volume surge
        short_signal = (price_below_weekly_ema and 
                       close[i] < l3_4h[i] * 0.999 and 
                       vol_ratio[i] > 1.8)
        
        # Exit: price crosses back through weekly EMA40 (regime change)
        exit_long = (position == 1 and close[i] <= ema40_1w_4h[i])
        exit_short = (position == -1 and close[i] >= ema40_1w_4h[i])
        
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