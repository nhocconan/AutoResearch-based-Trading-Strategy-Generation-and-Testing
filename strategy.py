#!/usr/bin/env python3
# 12h_1d_Camarilla_R3S3_Breakout_Volume
# Hypothesis: Trade 12h breakouts of daily Camarilla R3/S3 levels with volume confirmation and trend filter from 1d EMA50.
# The Camarilla R3/S3 levels represent strong intraday support/resistance; breakouts with volume indicate momentum.
# Daily EMA50 ensures alignment with the higher timeframe trend, reducing false signals in choppy markets.
# Designed for low frequency (15-30 trades/year) to survive both bull and bear markets by following higher timeframe structure.

name = "12h_1d_Camarilla_R3S3_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Daily Camarilla levels (R3, S3) ===
    # Calculate from daily OHLC (previous completed day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for current day's Camarilla
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Calculate Camarilla levels
    range_prev = high_prev - low_prev
    r3 = close_prev + range_prev * 1.1 / 2
    s3 = close_prev - range_prev * 1.1 / 2
    
    # Align daily levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === Volume confirmation (24-period average) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r3_aligned[i]
        breakout_down = close[i] < s3_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: breakout above R3, uptrend, volume confirmation
            if breakout_up and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: breakout below S3, downtrend, volume confirmation
            elif breakout_down and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: breakdown below S3 or trend reversal
            if breakout_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: breakout above R3 or trend reversal
            if breakout_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals