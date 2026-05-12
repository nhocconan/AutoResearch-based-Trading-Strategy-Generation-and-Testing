#!/usr/bin/env python3
# 1h_Camarilla_R1S1_Breakout_4hTrend_Volume
# Hypothesis: On 1h timeframe, enter long when price closes above 4h R1 with close > 4h EMA34 and volume > 1.5x average.
# Enter short when price closes below 4h S1 with close < 4h EMA34 and volume > 1.5x average.
# Exit when price crosses 4h EMA34 (trend reversal).
# Uses 4h for signal direction (trend + S/R levels) and 1h only for entry timing.
# Session filter: 08-20 UTC to avoid low-volume Asian session.
# Target: 15-37 trades/year (~60-150 total over 4 years) to minimize fee drag.
# Works in bull markets via breakouts and in bear via short reversals at S1.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Load 4h data for Camarilla pivot calculation and EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    # Calculate 4h pivot point and range
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # Camarilla R1 and S1 levels (4h)
    r1_4h = pivot_4h + range_4h * 1.083
    s1_4h = pivot_4h - range_4h * 1.083
    
    # 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume confirmation: 24-period moving average (1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready or outside session
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_4h_aligned[i]
        s1_val = s1_4h_aligned[i]
        ema4h_trend = ema34_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above 4h R1 with volume > 1.5x average and close > 4h EMA34
            if close[i] > r1_val and close[i] > ema4h_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # SHORT: Price closes below 4h S1 with volume > 1.5x average and close < 4h EMA34
            elif close[i] < s1_val and close[i] < ema4h_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 4h EMA34 (trend reversal)
            if close[i] < ema4h_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above 4h EMA34 (trend reversal)
            if close[i] > ema4h_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals