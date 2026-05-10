#!/usr/bin/env python3
# 6H_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike
# Hypothesis: Uses 6h timeframe with 1d for Camarilla pivot levels and 1w for trend confirmation.
# Enters long when price breaks above daily R3 with weekly trend up (close > EMA50) and volume > 2x 20-period average.
# Enters short when price breaks below daily S3 with weekly trend down (close < EMA50) and volume confirmation.
# Exits when price returns to opposite level (S3 for long, R3 for short) or weekly trend reverses.
# Uses weekly EMA50 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 12-37 trades per year on 6h timeframe with position size 0.25 to minimize fee drag.
# Includes minimum holding period of 2 bars to reduce churn.

name = "6H_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot_range = prev_high - prev_low
    r3_level = prev_close + 1.1 * pivot_range
    s3_level = prev_close - 1.1 * pivot_range
    
    # Align pivot levels to 6h timeframe (available after 1d bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume filter: volume > 2x 20-period average on 6h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Require minimum 2 bars since last exit to prevent churn
            if bars_since_entry >= 2:
                # Long entry: price breaks above R3 in uptrend with volume spike
                if (close[i] > r3_aligned[i] and 
                    price_above_ema and 
                    volume[i] > vol_threshold[i]):
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                # Short entry: price breaks below S3 in downtrend with volume spike
                elif (close[i] < s3_aligned[i] and 
                      price_below_ema and 
                      volume[i] > vol_threshold[i]):
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        elif position == 1:
            # Long exit: price returns to S3 or weekly trend reverses to downtrend
            if (close[i] < s3_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
                bars_since_entry += 1
        elif position == -1:
            # Short exit: price returns to R3 or weekly trend reverses to uptrend
            if (close[i] > r3_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
                bars_since_entry += 1
    
    return signals