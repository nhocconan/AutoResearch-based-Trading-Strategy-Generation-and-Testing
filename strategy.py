#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels from 1d timeframe to identify key S3/R3 levels for mean reversion entries and S4/R4 for breakout continuation. 
Combined with 1d EMA34 trend filter and volume spike confirmation. Designed for low frequency (15-30 trades/year) to minimize fee drag while capturing 
both mean reversion in ranges and breakout trends in trending markets. Works in bull/bear via adaptive entry logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 350:  # Need sufficient data for 1d indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # where C, H, L are from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First value has no previous day
    
    # Calculate Camarilla levels
    R4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    S4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to 6t timeframe
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 24-period average (approx 6d average)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(R4_6h[i]) or 
            np.isnan(S4_6h[i]) or np.isnan(ema_34_6h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = R3_6h[i]
        s3_val = S3_6h[i]
        r4_val = R4_6h[i]
        s4_val = S4_6h[i]
        ema_trend = ema_34_6h[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Mean reversion longs at S3 with volume spike in uptrend
            if close_val <= s3_val and close_val > ema_trend and vol_spike:
                signals[i] = size
                position = 1
            # Mean reversion shorts at R3 with volume spike in downtrend
            elif close_val >= r3_val and close_val < ema_trend and vol_spike:
                signals[i] = -size
                position = -1
            # Breakout longs above R4 with volume spike
            elif close_val >= r4_val and vol_spike:
                signals[i] = size
                position = 1
            # Breakout shorts below S4 with volume spike
            elif close_val <= s4_val and vol_spike:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 or above R4 (take profit)
            if close_val < s3_val or close_val > r4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above R3 or below S4 (take profit)
            if close_val > r3_val or close_val < s4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0