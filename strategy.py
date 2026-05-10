#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
Hypothesis: Use Camarilla pivot levels (from 1d) as key support/resistance. 
Long when price breaks above R1 in uptrend (12h EMA50 up) with volume confirmation.
Short when price breaks below S1 in downtrend (12h EMA50 down) with volume confirmation.
Includes volatility filter to avoid choppy markets. Designed for 4h timeframe.
Targets 20-50 trades per year to minimize fee drag. Works in both bull (breakout longs) 
and bear (breakdown shorts) by following 12h trend.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d volume SMA20 for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Daily Camarilla pivot levels (from 1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot points
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_factor = range_1d * 1.1
    r1_1d = close_1d + camarilla_factor / 12
    s1_1d = close_1d - camarilla_factor / 12
    
    # Align daily Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1)  # Need EMA50 and at least one daily pivot
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0  # 6x 4h periods in 1d
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        # Volatility filter: avoid extreme volatility that causes whipsaws
        if i >= 20:
            atr_20 = np.mean(np.maximum.reduce([
                high[i-19:i+1] - low[i-19:i+1],
                np.abs(high[i-19:i+1] - np.append([close[i-19]], close[i-19:i])),
                np.abs(low[i-19:i+1] - np.append([close[i-19]], close[i-19:i]))
            ])) if i >= 19 else 0
            vol_ma20 = np.mean([
                np.maximum.reduce([
                    high[max(0,i-39):i+1] - low[max(0,i-39):i+1],
                    np.abs(high[max(0,i-39):i+1] - np.append([close[max(0,i-39)]], close[max(0,i-39):i])),
                    np.abs(low[max(0,i-39):i+1] - np.append([close[max(0,i-39)]], close[max(0,i-39):i]))
                ])
            ]) if i >= 39 else atr_20
            volatility_filter = atr_20 < vol_ma20 * 1.5  # Avoid volatility spikes
        else:
            volatility_filter = True
        
        if position == 0:
            # Long: Price breaks above R1 in uptrend with volume confirmation
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume_confirm and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 in downtrend with volume confirmation
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume_confirm and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price falls below R1 or trend reversal
            if (close[i] < r1_1d_aligned[i] or 
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above S1 or trend reversal
            if (close[i] > s1_1d_aligned[i] or 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals