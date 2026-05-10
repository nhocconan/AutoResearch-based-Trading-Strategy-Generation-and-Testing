#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Price breaks Camarilla R3/S3 levels (from previous day) on 4h timeframe, confirmed by volume surge and 1d EMA34 trend. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out). Targets 25-40 trades/year to minimize fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R3 and S3 levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume average (6-period = 1.5 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Warmup: need EMA34 (34) + volume MA (6) + shifted Camarilla (1)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from 1d EMA34
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        uptrend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (1.8x average)
        volume_surge = volume[i] > 1.8 * vol_ma[i]
        
        # Breakout above R3 or breakdown below S3
        breakout_r3 = close[i] > camarilla_r3_aligned[i-1]
        breakdown_s3 = close[i] < camarilla_s3_aligned[i-1]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Breakout above R3 with volume surge and 1d uptrend
            if breakout_r3 and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S3 with volume surge and 1d downtrend
            elif breakdown_s3 and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 3 bars (6 hours)
            if bars_since_entry < 3:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below S3 or trend changes
                if close[i] < camarilla_s3_aligned[i-1] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price breaks above R3 or trend changes
                if close[i] > camarilla_r3_aligned[i-1] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals