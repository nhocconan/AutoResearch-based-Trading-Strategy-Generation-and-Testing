#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike
Hypothesis: Camarilla pivot levels (R3/S3) from 1d act as strong support/resistance. 
Break above R3 with volume spike and 1d uptrend = long. Break below S3 with volume spike 
and 1d downtrend = short. Uses 12h timeframe to reduce noise and overtrading.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R3, S3)
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 12h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Trend filter: 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA + 34 for EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and 1d uptrend
            if close[i] > camarilla_r3_aligned[i] and volume_confirm[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and 1d downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume_confirm[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price moves back below R3 OR 1d trend changes to downtrend
            if close[i] < camarilla_r3_aligned[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price moves back above S3 OR 1d trend changes to uptrend
            if close[i] > camarilla_s3_aligned[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0