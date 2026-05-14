#!/usr/bin/env python3
# 4h_12h_Camarilla_R3S3_Breakout_VolumeTrend
# Hypothesis: On 4h timeframe, trade breakouts from 12h-derived Camarilla R3/S3 levels with volume spike confirmation and 12h EMA trend filter.
# Uses 12h EMA34 to filter trades in trending markets. Targets 20-40 trades per year. Works in bull/bear via trend-aligned breakouts.
# Breakouts are confirmed by volume > 2x 20-period average and price beyond 0.5% buffer around R3/S3.
# Exits on reversal below/above S3/R3 or trend flip (price crosses 12h EMA34).

name = "4h_12h_Camarilla_R3S3_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (R3, S3)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for pivot calculation
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    
    # Pivot point and ranges
    pivot_12h = typical_price_12h
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R3 and S3
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    
    # 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R3, volume spike, and price above 12h EMA34 (uptrend)
            if (close[i] > r3_aligned[i] * 1.005 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S3, volume spike, and price below 12h EMA34 (downtrend)
            elif (close[i] < s3_aligned[i] * 0.995 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S3 or trend reversal (below EMA34)
            if close[i] < s3_aligned[i] * 0.995 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R3 or trend reversal (above EMA34)
            if close[i] > r3_aligned[i] * 1.005 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals