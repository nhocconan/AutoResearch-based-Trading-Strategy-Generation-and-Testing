#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Camarilla pivot levels from previous day (standard formula)
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    
    pivot = (c_high + c_low + c_close) / 3
    range_val = c_high - c_low
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Align pivot levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(c_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R3 in daily uptrend with volume
            if close[i] > r3_12h[i] and ema_34_12h[i] > ema_34_12h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in daily downtrend with volume
            elif close[i] < s3_12h[i] and ema_34_12h[i] < ema_34_12h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to pivot or trend reverses
            pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
            if close[i] < pivot_12h[i] or ema_34_12h[i] < ema_34_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to pivot or trend reverses
            pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
            if close[i] > pivot_12h[i] or ema_34_12h[i] > ema_34_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakouts with daily trend filter and volume confirmation on 12h timeframe
# - Camarilla R3/S3 represent strong support/resistance levels from previous day
# - Breakout above R3 in daily uptrend (EMA34 rising) signals bullish continuation
# - Breakdown below S3 in daily downtrend (EMA34 falling) signals bearish continuation
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price returns to pivot point or daily trend reverses
# - Position size 0.25 targets ~20-40 trades/year to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses 1d timeframe for structure and trend, 12h for execution timing
# - Similar 4h variants show strong test performance (Sharpe >1.8) with proper filtering
# - 12h timeframe reduces trade frequency while maintaining edge effectiveness
# - Focus on BTC and ETH as primary targets (avoid SOL-only bias)