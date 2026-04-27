#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_Breakout_1wTrend_Volume
Hypothesis: Use weekly-derived Camarilla R3/S3 levels as breakout triggers on 6h timeframe with 1-week EMA34 trend filter and volume spike (>2x 30-period average). Weekly timeframe provides stronger trend context than daily, reducing whipsaw in choppy markets. Target 15-30 trades/year to minimize fee decay while capturing major trend continuations. Works in bull (breakouts with weekly uptrend) and bear (mean reversion at extremes with weekly downtrend filter).
"""

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
    
    # Calculate weekly Camarilla levels (R3/S3) from 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Weekly Camarilla R3 and S3 levels (using standard 1.1 multiplier)
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align weekly Camarilla levels to 6h timeframe (wait for previous week's close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # 1-week EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume confirmation: current volume > 2.0 * 30-period average (higher threshold for 6h)
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for volume average and EMA
    start_idx = max(30, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation AND above weekly EMA34 (uptrend)
            if close[i] > camarilla_r3_val and vol_conf and close[i] > ema_34_val:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with volume confirmation AND below weekly EMA34 (downtrend)
            elif close[i] < camarilla_s3_val and vol_conf and close[i] < ema_34_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (opposite level)
            if close[i] < camarilla_s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R3 (opposite level)
            if close[i] > camarilla_r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0