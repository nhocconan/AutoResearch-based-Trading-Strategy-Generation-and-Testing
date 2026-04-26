#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation (>1.8x 24-period MA).
Long when price breaks above R3 in 1w uptrend with volume spike. Short when price breaks below S3 in 1w downtrend with volume spike.
Camarilla R3/S3 require stronger conviction to break (>1.1/4 range). Uses 1w trend to avoid counter-trend trades.
Designed for 12-37 trades/year on 12h timeframe. Works in bull/bear markets by following 1w trend. Discrete sizing 0.25 minimizes fee churn.
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
    
    # Get 1w data for Camarilla pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1w candle (avoid look-ahead)
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # where C, H, L are from prior completed 1w candle
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_r3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_s3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (they change only when new 1w candle forms)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    uptrend_1w = close > ema_50_1w_aligned
    downtrend_1w = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 1.8x 24-period MA (tight threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 24 for volume MA, 50 for EMA)
    start_idx = max(24, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 1w uptrend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                uptrend_1w[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with 1w downtrend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  downtrend_1w[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S3 (breakdown) OR 1w trend changes to downtrend
            if (close[i] < camarilla_s3_aligned[i] or not uptrend_1w[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R3 (breakout) OR 1w trend changes to uptrend
            if (close[i] > camarilla_r3_aligned[i] or not downtrend_1w[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0