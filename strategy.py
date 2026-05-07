#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # 4h data for trend filter (Camarilla R3/S3)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h: R3, S3
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R3 = close + 1.1*(high-low)/6
    # Camarilla S3 = close - 1.1*(high-low)/6
    camarilla_r3_4h = close_4h + 1.1 * (high_4h - low_4h) / 6
    camarilla_s3_4h = close_4h - 1.1 * (high_4h - low_4h) / 6
    
    # Align Camarilla levels to 1h
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Volume moving average on 1d
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 4h R3 with 1d volume spike
            if close[i] > r3_4h_aligned[i] and volume[i] > vol_ma_1d_aligned[i] * 1.5:
                signals[i] = 0.20
                position = 1
            # Short: break below 4h S3 with 1d volume spike
            elif close[i] < s3_4h_aligned[i] and volume[i] > vol_ma_1d_aligned[i] * 1.5:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price back below 4h S3
            if close[i] < s3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price back above 4h R3
            if close[i] > r3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and 1d volume confirmation
# - Uses 4h Camarilla levels (R3/S3) as breakout levels
# - Enters long when price breaks above 4h R3, short when breaks below 4h S3
# - Requires 1d volume spike (1.5x 20-day average) to confirm breakout strength
# - Uses session filter (08-20 UTC) to avoid low-volume Asian session noise
# - Exits when price returns to opposite Camarilla level (S3 for long, R3 for short)
# - Position size fixed at 0.20 to manage risk and reduce trade frequency
# - Works in both bull and bear markets by trading breakouts in direction of momentum
# - Targets 60-120 total trades over 4 years (15-30/year) to stay within limits
# - Camarilla levels provide mathematically derived support/resistance with statistical edge
# - Volume confirmation reduces false breakouts during low conviction moves
# - Session filter avoids overnight gaps and low liquidity periods