#!/usr/bin/env python3
# 12h_DailyCamarilla_Pivot_Breakout
# Hypothesis: Daily Camarilla pivot levels (R3/S3) with volume confirmation and 12h trend filter
# Works in bull markets via breakout momentum at R3 and in bear markets via breakdowns at S3
# Volume filter reduces false breakouts, trend filter avoids counter-trend trades
# Target: 15-35 trades per year (~60-140 over 4 years) with position size 0.25

name = "12h_DailyCamarilla_Pivot_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R3, S3)
    # Based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 2)  # R3 = pivot + 1.1*(H-L)/2
    s3 = pivot - (range_hl * 1.1 / 2)  # S3 = pivot - 1.1*(H-L)/2
    
    # Align daily pivot levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h EMA25 for trend filter
    ema_25_12h = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Volume ratio: current volume / 30-period average volume
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need 30 periods for volume MA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_25_12h[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions at Camarilla levels
        breakout_r3 = close[i] > r3_aligned[i]  # Break above R3
        breakdown_s3 = close[i] < s3_aligned[i]  # Break below S3
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        # Trend filter from 12h EMA25
        uptrend = close[i] > ema_25_12h[i]
        downtrend = close[i] < ema_25_12h[i]
        
        if position == 0:
            # Long: break above R3 + volume + uptrend
            if breakout_r3 and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 + volume + downtrend
            elif breakdown_s3 and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 or trend reversal to downtrend
            if close[i] < s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or trend reversal to uptrend
            if close[i] > r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals