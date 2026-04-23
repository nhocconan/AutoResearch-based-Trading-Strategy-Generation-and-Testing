#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (R3, S3) from 1w for breakout signals (stronger structure than 1d)
- 1w EMA50 as trend filter (long only above, short only below) for major trend alignment
- Volume > 2.0x 20-period average for confirmation (institutional participation)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts from weekly structure
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
    
    # 1w data for Camarilla pivot calculation (stronger structure)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels (R3, S3) from prior 1w bar
    pivot = (high_1w + low_1w + close_1w) / 3.0
    rng = high_1w - low_1w
    camarilla_r3 = pivot + (rng * 1.1 / 4.0)
    camarilla_s3 = pivot - (rng * 1.1 / 4.0)
    
    # Align Camarilla levels to 1w timeframe (using completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # 1w data for EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
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
            # Long: 1w Camarilla R3 breakout up AND price above 1w EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 1w Camarilla S3 breakout down AND price below 1w EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1w Camarilla S3 breakdown OR price crosses below 1w EMA50
            if breakout_down or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1w Camarilla R3 breakout OR price crosses above 1w EMA50
            if breakout_up or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike_Filter_v1"
timeframe = "1d"
leverage = 1.0