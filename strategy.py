#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_1wCamarilla_Breakout_v1
Hypothesis: On 1d timeframe, using Kaufman Adaptive Moving Average (KAMA) as a dynamic trend filter combined with weekly Camarilla R3/S3 breakouts and volume confirmation reduces whipsaws in ranging markets while capturing strong trends. KAMA adapts to market noise, making it effective in both bull and bear regimes by avoiding false breakouts during choppy periods. Volume confirmation ensures breakouts have conviction. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d KAMA for trend filter (adaptive to market noise)
    close_1d = prices['close'].values  # Use primary timeframe close for KAMA
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    total_change = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        total_change[i] = total_change[i-1] + np.abs(close_1d[i] - close_1d[i-1])
        if i >= 10:
            total_change[i] -= np.abs(close_1d[i-10] - close_1d[i-11])
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        net_change = np.abs(close_1d[i] - close_1d[i-10])
        er[i] = net_change / total_change[i] if total_change[i] != 0 else 0
    # Smoothing constants: fastest SC = 2/(2+1) = 0.666, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.666 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed value
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1w Camarilla levels (R3, S3) using previous 1w's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift to get previous week's close
    close_1w_shifted = np.concatenate([[np.nan], close_1w[:-1]])
    
    camarilla_range = high_1w - low_1w
    r3 = close_1w_shifted + 1.1 * camarilla_range / 4
    s3 = close_1w_shifted - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 1d volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for KAMA and volume MA)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (KAMA)
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # Weekly Camarilla breakout conditions
        breakout_r3 = close[i] > r3_aligned[i]
        breakout_s3 = close[i] < s3_aligned[i]
        
        # Long logic: breakout above R3 in uptrend with volume
        if uptrend and volume_spike and breakout_r3:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S3 in downtrend with volume
        elif downtrend and volume_spike and breakout_s3:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_Filter_1wCamarilla_Breakout_v1"
timeframe = "1d"
leverage = 1.0