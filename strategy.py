#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from 1d data provide institutional support/resistance.
Breakouts above R3 or below S3 with volume confirmation and 1d EMA34 trend filter capture
institutional breakout moves. Works in bull (breakouts above R3) and bear (breakdowns below S3).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Calculate Camarilla levels for each 1d bar: R3, S3, R4, S4
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # R4 = close + 1.1*(high-low)/1, S4 = close - 1.1*(high-low)/1
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            diff = high_1d[i] - low_1d[i]
            camarilla_r3[i] = close_1d[i] + 1.1 * diff / 2.0
            camarilla_s3[i] = close_1d[i] - 1.1 * diff / 2.0
            camarilla_r4[i] = close_1d[i] + 1.1 * diff
            camarilla_s4[i] = close_1d[i] - 1.1 * diff
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA and volume
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled to 6h)
        vol_6h_approx = vol_sma20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.5 * vol_6h_approx
        
        if position == 0:
            # Long: Break above R3 with uptrend and volume confirmation
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with downtrend and volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below S3 (failed breakout) or trend reversal
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above R3 (failed breakdown) or trend reversal
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals