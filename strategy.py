#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike"
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
    
    # Daily data for Camarilla pivot levels (1D)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h close for daily pivot calculation
    close_4h = pd.Series(close)
    
    # Daily OHLC for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # Previous day's values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    camarilla_r3_unadj = pivot + (range_val * 1.1 / 4)
    camarilla_s3_unadj = pivot - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 4h
    camarilla_r3 = align_htf_to_ltf(prices, df_1d, camarilla_r3_unadj)
    camarilla_s3 = align_htf_to_ltf(prices, df_1d, camarilla_s3_unadj)
    
    # Volume spike detection (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R3 in daily uptrend with volume spike
            if close[i] > camarilla_r3[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in daily downtrend with volume spike
            elif close[i] < camarilla_s3[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below pivot or trend reversal
            pivot_daily = (high_1d + low_1d + close_1d) / 3
            pivot_daily_aligned = align_htf_to_ltf(prices, df_1d, pivot_daily)
            if close[i] < pivot_daily_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above pivot or trend reversal
            pivot_daily = (high_1d + low_1d + close_1d) / 3
            pivot_daily_aligned = align_htf_to_ltf(prices, df_1d, pivot_daily)
            if close[i] > pivot_daily_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with daily trend filter and volume spike
# - Camarilla R3 (resistance) and S3 (support) derived from previous day's OHLC
# - Long when price breaks above R3 in daily uptrend (EMA34 rising) with volume spike (2x average)
# - Short when price breaks below S3 in daily downtrend (EMA34 falling) with volume spike
# - Exit when price returns to daily pivot or trend reverses
# - Uses 4h timeframe for execution with 1d Camarilla levels and trend filter
# - Volume spike (2x average) confirms breakout validity
# - Position size 0.25 targets ~25-50 trades/year to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Proven pattern from top performers: Camarilla + trend + volume spike is effective
# - Aims for 100-200 total trades over 4 years (25-50/year) within limits
# - Avoids overtrading by requiring multiple confluence factors (level break + trend + volume)