#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Signal_1wTrend_Volume
# Hypothesis: Trade reversals at Camarilla pivot levels on 12h with weekly trend filter and volume confirmation.
# Long when: price touches S1/S2 level during weekly uptrend with volume > 1.3x average.
# Short when: price touches R1/R2 level during weekly downtrend with volume > 1.3x average.
# Uses Camarilla levels from prior 1d for intraday support/resistance.
# Weekly trend filter avoids counter-trend trades in strong trends.
# Volume confirmation ensures institutional participation.
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years).

name = "12h_Camarilla_Pivot_Signal_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate Camarilla levels from prior 1d
    # Need prior day's OHLC for current 12h bar's levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # R2 = close + (high-low)*1.1/6
    # R1 = close + (high-low)*1.1/12
    # S1 = close - (high-low)*1.1/12
    # S2 = close - (high-low)*1.1/6
    # S3 = close - (high-low)*1.1/4
    # S4 = close - (high-low)*1.1/2
    
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12
    r2 = close_1d + range_1d * 1.1 / 6
    s1 = close_1d - range_1d * 1.1 / 12
    s2 = close_1d - range_1d * 1.1 / 6
    
    # Align Camarilla levels to 12h (prior day's levels for current bar)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = close_1w > ema20_1w
    weekly_downtrend = close_1w < ema20_1w
    
    # Align weekly trend to 12h
    weekly_uptrend_12h = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_12h = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(s2_12h[i]) or
            np.isnan(vol_ma[i]) or np.isnan(weekly_uptrend_12h[i]) or np.isnan(weekly_downtrend_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.3
        
        weekly_up = weekly_uptrend_12h[i] > 0.5
        weekly_down = weekly_downtrend_12h[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + price near S1/S2 + volume
            if weekly_up and volume_confirm:
                if (abs(close[i] - s1_12h[i]) / s1_12h[i] < 0.005 or 
                    abs(close[i] - s2_12h[i]) / s2_12h[i] < 0.005):
                    signals[i] = 0.25
                    position = 1
            # Enter short: weekly downtrend + price near R1/R2 + volume
            elif weekly_down and volume_confirm:
                if (abs(close[i] - r1_12h[i]) / r1_12h[i] < 0.005 or 
                    abs(close[i] - r2_12h[i]) / r2_12h[i] < 0.005):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions: trend weakens or price moves to opposite level
            if not weekly_up or \
               (abs(close[i] - r1_12h[i]) / r1_12h[i] < 0.005) or \
               (abs(close[i] - r2_12h[i]) / r2_12h[i] < 0.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend weakens or price moves to opposite level
            if not weekly_down or \
               (abs(close[i] - s1_12h[i]) / s1_12h[i] < 0.005) or \
               (abs(close[i] - s2_12h[i]) / s2_12h[i] < 0.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals