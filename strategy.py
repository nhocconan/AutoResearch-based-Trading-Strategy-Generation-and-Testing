#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_Spike"
timeframe = "4h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate daily Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Handle first value
    close_1d_prev[0] = close_1d[0]
    
    # Camarilla formula: Range = High - Low
    # R3 = Close + 1.1 * (High - Low) / 6
    # S3 = Close - 1.1 * (High - Low) / 6
    daily_range = high_1d - low_1d
    r3 = close_1d_prev + 1.1 * daily_range / 6
    s3 = close_1d_prev - 1.1 * daily_range / 6
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with volume and in uptrend
            vol_condition = volume[i] > vol_ma[i] * 1.5
            uptrend = close[i] > ema_34_aligned[i]
            
            if close[i] > r3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and in downtrend
            elif close[i] < s3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA34 or volume drops
            if close[i] < ema_34_aligned[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA34 or volume drops
            if close[i] > ema_34_aligned[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with daily EMA34 trend filter and volume spike confirmation.
# Uses daily Camarilla levels (R3/S3) as key support/resistance from previous day.
# Breaks above R3 with volume in uptrend = long, breaks below S3 with volume in downtrend = short.
# Daily EMA34 ensures trades align with higher timeframe trend.
# Volume spike (>1.5x average) confirms institutional participation.
# Works in bull (buy R3 breaks in uptrend) and bear (sell S3 breaks in downtrend).
# Position size 0.25 balances risk and keeps trade frequency ~15-30/year.