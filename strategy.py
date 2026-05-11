#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

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
    
    # 1. Load weekly data ONCE for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    
    # 2. Weekly EMA200 for trend filter (weekly timeframe)
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 3. Calculate weekly high/low/close for pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 4. Standard pivot point system (weekly)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # 5. Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 6. Volume filter: 24-period EMA for spike detection (4 days of 6h bars)
    vol_ema24 = pd.Series(volume).ewm(span=24, min_periods=24, adjust=False).mean().values
    volume_ok = volume > vol_ema24 * 2.0  # Require strong volume spike
    
    # 7. Fixed position size to avoid churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema200 = close[i] > ema200_1w_aligned[i]
        price_below_ema200 = close[i] < ema200_1w_aligned[i]
        breakout_long = close[i] > r3_aligned[i]  # Break above R3 for strong bullish
        breakout_short = close[i] < s3_aligned[i]  # Break below S3 for strong bearish
        
        if position == 0:
            # Long: Price breaks above R3 + above weekly EMA200 + volume spike
            if breakout_long and price_above_ema200 and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below weekly EMA200 + volume spike
            elif breakout_short and price_below_ema200 and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - use pivot as mean reversion target
            if position == 1:
                # Exit: Price crosses below pivot OR trend reverses
                if close[i] < pivot_aligned[i] or close[i] < ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above pivot OR trend reverses
                if close[i] > pivot_aligned[i] or close[i] > ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals