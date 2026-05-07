#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Load daily data ONCE for Camarilla and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2x 20-period average
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: breakout above R3 with volume in uptrend
            if close[i] > R3_aligned[i] and vol_condition and ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S3 with volume in downtrend
            elif close[i] < S3_aligned[i] and vol_condition and ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: reverse signal or trend change
            if close[i] < S3_aligned[i] or ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: reverse signal or trend change
            if close[i] > R3_aligned[i] or ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakouts with daily trend filter and volume confirmation
# - Camarilla R3/S3 act as key support/resistance levels derived from previous day's range
# - Breakout above R3 with volume indicates bullish momentum continuation
# - Breakdown below S3 with volume indicates bearish momentum continuation
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (2x average) reduces false breakouts
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Position size 0.25 targets ~50-150 trades over 4 years (12-37/year) to stay within limits
# - Proven pattern: similar variants show strong performance in ETH/SOL (see DB top performers)
# - Novel for 6h timeframe: combines Camarilla levels with trend and volume filters not recently tested at this TF