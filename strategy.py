#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
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
    
    # Load daily data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection: 3-period average (3x4h = 12h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 3)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S3 with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_3[i] * 2.0
            uptrend = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            
            if close[i] > s3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price below R3 with volume and 12h downtrend
            elif close[i] < r3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below S3 or volume drops or trend reversal
            if close[i] < s3_aligned[i] or volume[i] < vol_ma_3[i] * 1.2 or ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above R3 or volume drops or trend reversal
            if close[i] > r3_aligned[i] or volume[i] < vol_ma_3[i] * 1.2 or ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend and volume confirmation
# - Camarilla R3/S3 are strong support/resistance levels from previous day
# - Breakout above S3 with volume in 12h uptrend = high-probability long
# - Breakdown below R3 with volume in 12h downtrend = high-probability short
# - Volume spike (2.0x 3-period average) confirms institutional participation
# - 12h EMA50 trend filter reduces whipsaws and aligns with higher timeframe momentum
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to S3/R3, volume weakens, or trend reverses
# - Position size 0.30 targets ~20-50 trades/year, avoiding excessive fee drag
# - Uses actual daily Camarilla levels (not intraday) for better stability
# - Novel combination: Daily Camarilla (1d) + trend (12h) + volume (4h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Focus on BTC/ETH as primary targets, avoiding SOL-only bias