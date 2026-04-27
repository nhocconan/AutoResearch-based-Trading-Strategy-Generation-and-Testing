#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot R3/S3 breakout with 1d trend filter (EMA50) and volume spike on 12h timeframe.
Camarilla levels provide institutional support/resistance. R3/S3 breaks indicate strong momentum.
Combined with 1d EMA50 trend filter to align with higher timeframe direction and volume surge for confirmation.
Designed for 50-150 trades over 4 years. Works in bull via breakouts above R3 in uptrend, bear via breakdowns below S3 in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike detector (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        r3 = R3_12h[i]
        s3 = S3_12h[i]
        ema = ema_50_12h[i]
        vol_spike_now = vol_spike[i]
        
        if position == 0:
            # Long: Break above R3 AND above 1d EMA50 AND volume spike
            if close[i] > r3 and close[i] > ema and vol_spike_now:
                signals[i] = size
                position = 1
            # Short: Break below S3 AND below 1d EMA50 AND volume spike
            elif close[i] < s3 and close[i] < ema and vol_spike_now:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Price drops below 1d EMA50 OR breaks below S3 (failure)
            if close[i] < ema or close[i] < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price rises above 1d EMA50 OR breaks above R3 (failure)
            if close[i] > ema or close[i] > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0