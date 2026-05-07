# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily OHLC (previous day's values)
    # Camarilla formula: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each day
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (using previous day's values)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 20-period average (approx 10 hours for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S3 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > S3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R3 with volume and daily downtrend
            elif close[i] < R3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to S3 or volume drops significantly
            if close[i] < S3_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to R3 or volume drops significantly
            if close[i] > R3_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with daily trend and volume confirmation
# - Camarilla R3 and S3 act as key support/resistance levels derived from daily price action
# - Breakout above S3 with volume expansion in daily uptrend = long opportunity
# - Breakdown below R3 with volume expansion in daily downtrend = short opportunity
# - Volume spike (2x average) confirms institutional participation and reduces false breakouts
# - Daily EMA(34) trend filter ensures we trade with the higher timeframe momentum
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to the breakout level (S3/R3) or volume weakens
# - Position size 0.25 targets 20-40 trades/year, avoiding excessive fee drag
# - Uses actual Camarilla formula rather than pivot points for better precision in ranging markets