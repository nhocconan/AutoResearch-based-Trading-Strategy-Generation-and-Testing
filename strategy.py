#!/usr/bin/env python3
"""
12h_1w_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Trade breakouts at weekly Camarilla R3/S3 levels on 12h timeframe with daily trend filter and volume confirmation.
Weekly Camarilla levels act as strong support/resistance. Breakouts in direction of daily trend with volume should capture
significant moves while avoiding false breakouts in ranging markets. Designed for low trade frequency to minimize fee drag.
"""

name = "12h_1w_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # === Weekly OHLC for Camarilla Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Weekly Camarilla Pivot Points from previous week's OHLC
    ph_w = df_1w['high'].values
    pl_w = df_1w['low'].values
    pc_w = df_1w['close'].values
    
    # Weekly Pivot Point (PP)
    pp_w = (ph_w + pl_w + pc_w) / 3.0
    # Weekly R3 and S3 (stronger breakout levels than R1/S1)
    r3_w = pp_w + (ph_w - pl_w) * 1.1
    s3_w = pp_w - (ph_w - pl_w) * 1.1
    
    # Align to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1w, r3_w)
    s3_12h = align_htf_to_ltf(prices, df_1w, s3_w)
    pp_12h = align_htf_to_ltf(prices, df_1w, pp_w)
    
    # === Daily Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Filter (2.0x 30-period EMA on 12h) ===
    vol_ema30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_ok = volume > vol_ema30 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly and daily calculations)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(ema34_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R3 with uptrend and volume
            if (close[i] > r3_12h[i] and 
                close[i] > ema34_12h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S3 with downtrend and volume
            elif (close[i] < s3_12h[i] and 
                  close[i] < ema34_12h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below weekly pivot (mean reversion to pivot)
            if close[i] < pp_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above weekly pivot (mean reversion to pivot)
            if close[i] > pp_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals