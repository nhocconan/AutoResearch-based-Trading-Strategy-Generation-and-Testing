#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Trend_Filter_v1
Hypothesis: Breakout of daily Camarilla H4/L4 levels with 1d trend filter (price > EMA50) and volume confirmation.
Works in bull markets (breakouts continue) and bear markets (false breakouts fade quickly via trend filter).
Target: 20-50 trades per year (~80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Trend_Filter_v1"
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
    
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        range_val = high_1d[i] - low_1d[i]
        camarilla_h4[i] = close_1d[i] + range_val * 1.1 / 4
        camarilla_l4[i] = close_1d[i] - range_val * 1.1 / 4
    
    # Align to 4h timeframe
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === DAILY TREND FILTER (EMA50) ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    trend_up = close > ema50_4h
    trend_down = close < ema50_4h
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(h4_4h[i]) or np.isnan(l4_4h[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = (close[i] > h4_4h[i] and trend_up[i] and vol_ratio[i] > 1.5)
        breakout_down = (close[i] < l4_4h[i] and trend_down[i] and vol_ratio[i] > 1.5)
        
        # Exit conditions
        exit_long = position == 1 and (close[i] < ema50_4h[i] or close[i] < l4_4h[i])
        exit_short = position == -1 and (close[i] > ema50_4h[i] or close[i] > h4_4h[i])
        
        # Execute trades
        if breakout_up and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and position != -1:
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