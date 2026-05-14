#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (R3/S3) from 1d for breakout signals
- 12h EMA50 as trend filter (long only above, short only below)
- Volume > 2.0x 20-period average for confirmation
- Position size: 0.25 discrete level to minimize fee churn
- Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts
"""

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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R3, S3) from prior 1d bar
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    hl_range = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * hl_range / 2.0
    camarilla_s3 = close_1d - 1.1 * hl_range / 2.0
    
    # Align Camarilla levels to 4h timeframe (using completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_r3_aligned[i]  # Close above R3
        breakout_down = close[i] < camarilla_s3_aligned[i]  # Close below S3
        
        if position == 0:
            # Long: 4h Camarilla R3 breakout up AND price above 12h EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 4h Camarilla S3 breakout down AND price below 12h EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 4h Camarilla S3 breakdown OR price crosses below 12h EMA50
            if breakout_down or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 4h Camarilla R3 breakout OR price crosses above 12h EMA50
            if breakout_up or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_Filter_v1"
timeframe = "4h"
leverage = 1.0