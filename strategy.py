#!/usr/bin/env python3
name = "6h_12h_1d_Camarilla_R3S3_Breakout_Trend"
timeframe = "6h"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels (R3 and S3 for breakouts)
    s3 = prev_close - (range_hl * 1.1000 / 2)  # S3 level
    r3 = prev_close + (range_hl * 1.1000 / 2)  # R3 level
    
    # Align daily levels to 6h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # 12h trend filter: EMA(34) on 12h close
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above R3 with volume and 12h uptrend (breakout)
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]
            
            if close[i] > r3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below S3 with volume and 12h downtrend (breakdown)
            elif close[i] < s3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below R3 or volume drops
            if close[i] < r3_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above S3 or volume drops
            if close[i] > s3_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h trend and volume confirmation
# - Daily Camarilla R3/S3 act as strong breakout/breakdown levels (wider than R1/S1)
# - Breakout above R3 with volume in 12h uptrend = long opportunity
# - Breakdown below S3 with volume in 12h downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Uses 12h EMA(34) for trend filter to avoid whipsaws in choppy markets
# - Works in both bull (buy R3 breaks in uptrend) and bear (sell S3 breaks in downtrend)
# - Exit when price returns to R3/S3 or volume weakens
# - Position size 0.25 targets ~30-60 trades/year, avoiding fee drag
# - Novelty: Using R3/S3 levels (not commonly tested) with 12h trend filter on 6h timeframe