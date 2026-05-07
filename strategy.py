#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Load daily data ONCE for Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    weekly_close = df_1w['close'].values
    ema_21_1w = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily Camarilla pivot levels from previous day
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    range_val = daily_high - daily_low
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume spike detection (2x 20-period average on 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_21_12h[i]) or np.isnan(pivot_12h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R3 in weekly uptrend with volume
            if close[i] > r3_12h[i] and ema_21_12h[i] > ema_21_12h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in weekly downtrend with volume
            elif close[i] < s3_12h[i] and ema_21_12h[i] < ema_21_12h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to pivot or weekly trend reverses
            if close[i] < pivot_12h[i] or ema_21_12h[i] < ema_21_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to pivot or weekly trend reverses
            if close[i] > pivot_12h[i] or ema_21_12h[i] > ema_21_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakouts on 12h with weekly trend filter and volume confirmation
# - Weekly trend filter ensures we trade with the higher timeframe momentum
# - Camarilla R3/S3 represent strong daily support/resistance levels
# - Breakout above R3 in weekly uptrend signals bullish continuation
# - Breakdown below S3 in weekly downtrend signals bearish continuation
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price returns to daily pivot or weekly trend reverses
# - Position size 0.25 targets ~20-40 trades/year to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses 1w timeframe for trend and 1d for structure, 12h for execution
# - Lower frequency reduces noise and improves signal quality in choppy markets