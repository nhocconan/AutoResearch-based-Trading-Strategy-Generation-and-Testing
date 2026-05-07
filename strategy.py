#!/usr/bin/env python3
name = "12h_Camarilla_Pivot_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Load daily data ONCE for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily data
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = Pivot + (Range * 1.1 / 12)
    # S1 = Pivot - (Range * 1.1 / 12)
    # R3 = Pivot + (Range * 1.1 / 4)
    # S3 = Pivot - (Range * 1.1 / 4)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    daily_range = daily_high - daily_low
    r1 = pivot + (daily_range * 1.1 / 12)
    s1 = pivot - (daily_range * 1.1 / 12)
    r3 = pivot + (daily_range * 1.1 / 4)
    s3 = pivot - (daily_range * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: Break above S1 with volume in daily uptrend
            if close[i] > s1_aligned[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.30
                position = 1
            # Short: Break below R1 with volume in daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: Price returns to pivot or trend reverses
            if close[i] < pivot[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price returns to pivot or trend reverses
            if close[i] > pivot[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Camarilla pivot breakouts on 12h with daily trend filter and volume confirmation
# - Uses Camarilla R1/S1 as entry levels (tighter than R3/S3 for higher probability)
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (2x average) reduces false breakouts
# - Works in both bull (buy S1 breakouts in uptrend) and bear (sell R1 breakdowns in downtrend)
# - Exit when price returns to daily pivot or trend reverses
# - Position size 0.30 targets ~40-100 trades over 4 years (10-25/year) to avoid fee drag
# - Camarilla levels provide mathematically derived support/resistance with statistical edge
# - Daily timeframe for pivots/trend avoids noise while 12h provides timely execution
# - Proven pattern: Similar to top performers (e.g., 4H_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Dyn) but adapted to 12h timeframe
# - Aims for 50-150 total trades over 4 years (12-37/year) as specified in experiment #135745
# - Avoids overtrading by requiring volume spike and trend alignment for entry
# - Simple 3-condition logic: pivot level break + volume + trend filter