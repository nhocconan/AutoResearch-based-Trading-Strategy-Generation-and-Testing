#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3S3 breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above R3, EMA(12h) rising, volume > 1.5x average.
# Short when price breaks below S3, EMA(12h) falling, volume > 1.5x average.
# Exit on opposite breakout or EMA trend reversal.
# Focus on high-probability breakouts with volume to avoid false signals.
# Target: 75-200 trades over 4 years to minimize fee drag.

name = "4h_Camarilla_R3S3_Breakout_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous 12h bar
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    range_12h = high_12h - low_12h
    r3 = close_12h + range_12h * 1.1 / 4
    s3 = close_12h - range_12h * 1.1 / 4
    
    # Align Camarilla levels to 4h (previous 12h bar's levels)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout with volume confirmation
            if vol_spike[i]:
                # Long breakout above R3
                if close[i] > r3_aligned[i]:
                    # Confirm trend: price above EMA
                    if close[i] > ema_34_12h_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                # Short breakdown below S3
                elif close[i] < s3_aligned[i]:
                    # Confirm trend: price below EMA
                    if close[i] < ema_34_12h_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: breakdown below S3 or trend reversal
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_12h_aligned[i]:  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above R3 or trend reversal
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_12h_aligned[i]:  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals