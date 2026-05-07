#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily close for Camarilla calculation
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_range = daily_high - daily_low
    r3 = daily_close + camarilla_range * 1.1 / 2
    s3 = daily_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 12h chart
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: 2x average volume
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: price breaks above R3 with volume in daily uptrend
            if close[i] > r3_aligned[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume in daily downtrend
            elif close[i] < s3_aligned[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below S3 (mean reversion) or trend breaks
            if close[i] < s3_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above R3 (mean reversion) or trend breaks
            if close[i] > r3_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with daily trend filter and volume confirmation
# - Camarilla R3/S3 levels act as natural support/resistance derived from prior day's range
# - Breakout above R3 (bullish) or below S3 (bearish) with volume continuation signals
# - Daily EMA34 trend filter ensures alignment with higher timeframe direction
# - Volume confirmation (2x average) reduces false breakouts in choppy markets
# - Mean reversion exit when price returns to opposite S3/R3 level or trend breaks
# - Works in both bull and breakout markets (breakouts) and bear markets (mean reversion)
# - Position size 0.25 targets ~50-150 trades over 4 years (12-37/year) to avoid fee drag
# - Proven pattern: Camarilla breakouts + volume + trend filter showed strong performance in DB
# - 12h timeframe reduces noise vs lower timeframes while capturing significant moves