#!/usr/bin/env python3
name = "1d_1w_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "1d"
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
    
    # Load weekly data ONCE for trend filter and volatility context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data ONCE for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Daily Camarilla pivot levels from previous day
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    pivot = (d_high + d_low + d_close) / 3
    range_val = d_high - d_low
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to daily timeframe
    r3_d = align_htf_to_ltf(prices, df_1d, r3)
    s3_d = align_htf_to_ltf(prices, df_1d, s3)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike detection (2x 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_d[i]) or np.isnan(s3_d[i]) or 
            np.isnan(ema_20_1w_d[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R3 in weekly uptrend with volume
            if close[i] > r3_d[i] and ema_20_1w_d[i] > ema_20_1w_d[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in weekly downtrend with volume
            elif close[i] < s3_d[i] and ema_20_1w_d[i] < ema_20_1w_d[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to pivot or weekly trend reverses
            pivot_d = align_htf_to_ltf(prices, df_1d, pivot)
            if close[i] < pivot_d[i] or ema_20_1w_d[i] < ema_20_1w_d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to pivot or weekly trend reverses
            pivot_d = align_htf_to_ltf(prices, df_1d, pivot)
            if close[i] > pivot_d[i] or ema_20_1w_d[i] > ema_20_1w_d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly trend-filtered Camarilla R3/S3 breakouts on daily chart
# - Camarilla R3/S3 represent strong support/resistance levels from previous day
# - Breakout above R3 in weekly uptrend (EMA20 rising) signals bullish continuation
# - Breakdown below S3 in weekly downtrend (EMA20 falling) signals bearish continuation
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price returns to pivot point or weekly trend reverses
# - Position size 0.25 targets ~15-25 trades/year to avoid fee drag
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses daily timeframe for execution and weekly for trend context
# - Similar Camarilla variants show strong test performance with proper filtering
# - Target: 30-100 total trades over 4 years (7-25/year) as specified for 1d timeframe