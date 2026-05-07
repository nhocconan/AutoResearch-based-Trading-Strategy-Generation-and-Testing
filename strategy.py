#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (R3, S3)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    cam_r3 = prev_close + (prev_high - prev_low) * 1.1 / 6
    cam_s3 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R3 in daily uptrend with volume spike
            if close[i] > cam_r3_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in daily downtrend with volume spike
            elif close[i] < cam_s3_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below R3 or trend change
            if close[i] < cam_r3_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above S3 or trend change
            if close[i] > cam_s3_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakouts with daily trend filter and volume confirmation
# - Camarilla R3/S3 are significant intraday resistance/support levels
# - Breakouts above R3 in uptrend or below S3 in downtrend with volume spike indicate strong momentum
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (2x average) reduces false breakouts
# - Works in both bull (R3 breakouts in uptrend) and bear (S3 breakdowns in downtrend)
# - Exit when price returns to the breakout level or trend changes
# - Position size 0.25 targets ~50-150 total trades over 4 years (12-37/year) to stay within limits
# - Camarilla levels provide clear structure with defined support/resistance levels
# - Daily trend filter reduces whipsaws vs same-timeframe signals
# - Proven combination: Camarilla breakout + trend + volume spike has worked well on ETH/SOL in 4h/6h
# - Adapting to 12h timeframe to reduce trade frequency and improve test generalization