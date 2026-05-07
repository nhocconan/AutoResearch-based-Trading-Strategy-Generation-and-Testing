#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Load 1d data ONCE for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla levels: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We'll use previous day's high/low/close for today's levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each day
    cam_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    cam_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    cam_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    cam_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or 
            np.isnan(cam_r4_aligned[i]) or np.isnan(cam_s4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume, in uptrend
            if close[i] > cam_r3_aligned[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with volume, in downtrend
            elif close[i] < cam_s3_aligned[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to S3 or trend changes
            if close[i] < cam_s3_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to R3 or trend changes
            if close[i] > cam_r3_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakouts with 1d trend filter and volume confirmation
# - Camarilla R3/S3 act as key support/resistance levels derived from previous day's range
# - Breakouts above R3 or below S3 with volume indicate institutional participation
# - 1d EMA34 trend filter ensures we only trade in the direction of higher timeframe trend
# - Volume confirmation (1.5x average) reduces false breakouts
# - Works in both bull (long breakouts in uptrend) and bear (short breakdowns in downtrend)
# - Exit when price returns to opposite Camarilla level (S3 for longs, R3 for shorts) or trend changes
# - Position size 0.25 targets ~50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Novel application: Camarilla levels (typically used intraday) applied to 6h with daily trend filter
# - Aims for 80-120 total trades over 4 years (20-30/year) to stay within limits for 6s timeframe