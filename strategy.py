#!/usr/bin/env python3
# 1d_Camarilla_R1S1_Breakout_1wEMA50_TrendFilter
# Hypothesis: On 1d timeframe, enter long when price closes above weekly R1 with close > weekly EMA50.
# Enter short when price closes below weekly S1 with close < weekly EMA50.
# Exit when price crosses weekly EMA50 (trend reversal).
# Uses weekly EMA for trend filter to reduce whipsaw and improve performance in both bull and bear markets.
# Targets 15-25 trades/year for low fee drift.

name = "1d_Camarilla_R1S1_Breakout_1wEMA50_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Camarilla pivot calculation and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot point and range
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Camarilla R1 and S1 levels
    r1 = weekly_pivot + weekly_range * 1.083
    s1 = weekly_pivot - weekly_range * 1.083
    
    # Calculate weekly EMA50
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly levels and weekly EMA50 to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema1w_trend = ema50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above R1 with close > weekly EMA50 and volume > 20MA
            if close[i] > r1_val and close[i] > ema1w_trend and volume[i] > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S1 with close < weekly EMA50 and volume > 20MA
            elif close[i] < s1_val and close[i] < ema1w_trend and volume[i] > vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly EMA50 (trend reversal)
            if close[i] < ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly EMA50 (trend reversal)
            if close[i] > ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals