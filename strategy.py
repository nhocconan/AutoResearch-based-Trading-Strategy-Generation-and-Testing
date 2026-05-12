#!/usr/bin/env python3
# 1d_WeeklyPivot_Breakout_WeeklyTrend_VolumeSpike
# Hypothesis: On 1d timeframe, enter long when price closes above weekly S2 with price > weekly EMA50 and volume spike.
# Enter short when price closes below weekly R2 with price < weekly EMA50 and volume spike.
# Exit when price crosses weekly EMA50 (trend reversal).
# Uses weekly timeframe for pivot levels and trend filter to avoid short-term noise.
# Targets 10-25 trades/year for low fee drift and works in both bull and bear markets by fading extreme weekly levels.

name = "1d_WeeklyPivot_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "1d"
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
    
    # Load weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot point and range
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Weekly R2 and S2 levels (stronger levels for breakout)
    r2 = weekly_pivot + weekly_range * 1.1000 / 4.0
    s2 = weekly_pivot - weekly_range * 1.1000 / 4.0
    
    # Weekly EMA50 for trend filter
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period moving average on daily data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to daily timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        weekly_trend = weekly_ema50_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above S2 with price > weekly EMA50 and volume > 1.5x MA
            if close[i] > s2_val and close[i] > weekly_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below R2 with price < weekly EMA50 and volume > 1.5x MA
            elif close[i] < r2_val and close[i] < weekly_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly EMA50 (trend reversal)
            if close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly EMA50 (trend reversal)
            if close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals