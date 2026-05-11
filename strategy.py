#!/usr/bin/env python3
"""
6H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: Uses 12h timeframe for trend filter (more stable than 1d) and 6h for entry timing. 
Buys when price breaks above 12h Camarilla R3 with 12h uptrend and 6h volume spike. 
Sells when price breaks below 12h S3 with 12h downtrend and volume spike. 
Focuses on 12h Camarilla R3/S3 levels (stronger than R1/S1) to reduce whipsaw in ranging markets.
Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
"""

name = "6H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h Camarilla levels: R3, S3 (outer levels for stronger signals)
    hl_range_12h = high_12h - low_12h
    r3_12h = close_12h + hl_range_12h * 1.5000
    s3_12h = close_12h - hl_range_12h * 1.5000
    
    # Align 12h Camarilla levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # 6h volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Fixed position size to minimize churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema12h = close[i] > ema50_12h_aligned[i]
        price_below_ema12h = close[i] < ema50_12h_aligned[i]
        breakout_long = close[i] > r3_12h_aligned[i]
        breakout_short = close[i] < s3_12h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above 12h EMA50 + volume spike
            if breakout_long and price_above_ema12h and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below 12h EMA50 + volume spike
            elif breakout_short and price_below_ema12h and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price crosses below S3 OR trend reverses (close below 12h EMA)
                if close[i] < s3_12h_aligned[i] or close[i] < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R3 OR trend reverses (close above 12h EMA)
                if close[i] > r3_12h_aligned[i] or close[i] > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals