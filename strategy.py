#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Load daily data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R3, S3)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla R3 and S3 levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    
    # Align daily Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S3 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_6[i] * 2.0
            uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if close[i] > s3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R3 with volume and weekly downtrend
            elif close[i] < r3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S3 or volume drops
            if close[i] < s3_aligned[i] or volume[i] < vol_ma_6[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R3 or volume drops
            if close[i] > r3_aligned[i] or volume[i] < vol_ma_6[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 1w trend and volume confirmation
# - Camarilla R3/S3 act as key support/resistance levels from prior day
# - Breakout above S3 with volume in weekly uptrend = long opportunity
# - Breakdown below R3 with volume in weekly downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Weekly trend filter reduces whipsaws vs using same timeframe
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to S3/R3 or volume weakens
# - Position size 0.25 targets ~25-40 trades/year, avoiding fee drag
# - Uses actual daily Camarilla levels for precision
# - Weekly trend filter provides multi-timeframe alignment
# - Volume confirmation reduces false breakouts
# - Designed to work in BOTH bull and bear markets via trend filter
# - Aims for 100-160 total trades over 4 years (25-40/year) to stay within limits