#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_25"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous 1d
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # R3 and S3 levels: Close +- 1.1 * (High - Low)
    r3 = prev_close + 1.1 * (prev_high - prev_low)
    s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and Camarilla
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with daily uptrend and volume
            if (close[i] > r3_aligned[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with daily downtrend and volume
            elif (close[i] < s3_aligned[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S3 or trend change
            if close[i] < s3_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 or trend change
            if close[i] > r3_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume confirmation.
# Camarilla levels identify key support/resistance from prior day. Breaking R3/S3 indicates strong momentum.
# Daily EMA filter ensures alignment with higher timeframe trend. Volume confirms institutional participation.
# Target: 25-35 trades/year to minimize fee drift. Position size 0.25 limits drawdown in volatile markets.