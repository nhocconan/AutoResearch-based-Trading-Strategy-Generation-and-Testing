#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot level breakout with volume confirmation and 1-day EMA trend filter.
- Camarilla levels (R1-S1) act as support/resistance; breakouts indicate momentum
- Volume spike confirms institutional participation (volume > 1.5x 20-period avg)
- 1-day EMA(50) filters for trend direction: only long above EMA, short below
- Exit on opposite Camarilla level touch or volume dry-up
- Target: 25-40 trades/year to avoid fee drag
- Uses discrete position sizing (0.25) to minimize churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    camarilla_r1 = np.full(len(high_1d), np.nan)
    camarilla_s1 = np.full(len(high_1d), np.nan)
    camarilla_r2 = np.full(len(high_1d), np.nan)
    camarilla_s2 = np.full(len(high_1d), np.nan)
    camarilla_r3 = np.full(len(high_1d), np.nan)
    camarilla_s3 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        # Camarilla formulas
        camarilla_r1[i] = prev_close + range_ * 1.1 / 12
        camarilla_s1[i] = prev_close - range_ * 1.1 / 12
        camarilla_r2[i] = prev_close + range_ * 1.1 / 6
        camarilla_s2[i] = prev_close - range_ * 1.1 / 6
        camarilla_r3[i] = prev_close + range_ * 1.1 / 4
        camarilla_s3[i] = prev_close - range_ * 1.1 / 4
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1-day EMA(50)
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = close_1d[i] * 0.0377 + ema_50[i-1] * (1 - 0.0377)  # alpha = 2/(50+1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(30, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 + volume spike + price above EMA50
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + volume spike + price below EMA50
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price touches S1 OR volume dry-up (volume < 0.5 * MA20)
            volume_dry = volume[i] < (0.5 * vol_ma_20[i]) if not np.isnan(vol_ma_20[i]) else False
            if (close[i] <= camarilla_s1_aligned[i] or 
                volume_dry):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches R1 OR volume dry-up
            volume_dry = volume[i] < (0.5 * vol_ma_20[i]) if not np.isnan(vol_ma_20[i]) else False
            if (close[i] >= camarilla_r1_aligned[i] or 
                volume_dry):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_VolumeSpike_EMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0