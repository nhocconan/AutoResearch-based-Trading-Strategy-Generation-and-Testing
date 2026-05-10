#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Daily Camarilla R3/S3 breakouts with weekly trend filter and volume surge.
# Works in bull/bear by requiring trend alignment, reducing false breakouts. Targets 10-25 trades/year.

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (R3, S3)
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_r3 = close + (high - low) * 1.1 / 2
    camarilla_s3 = close - (high - low) * 1.1 / 2
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume average (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: need weekly EMA34 (34) + volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from weekly EMA34
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
        uptrend = close_1w_aligned[i] > ema_34_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation (2x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # Breakout above R3 or breakdown below S3
        breakout_r3 = close[i] > camarilla_r3[i-1]
        breakdown_s3 = close[i] < camarilla_s3[i-1]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Breakout above R3 with volume surge and weekly uptrend
            if breakout_r3 and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S3 with volume surge and weekly downtrend
            elif breakdown_s3 and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 3 days
            if bars_since_entry < 3:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below S3 or trend changes
                if close[i] < camarilla_s3[i-1] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price breaks above R3 or trend changes
                if close[i] > camarilla_r3[i-1] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals