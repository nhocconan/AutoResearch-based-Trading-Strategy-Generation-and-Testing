#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: On 6h timeframe, price breaks above Camarilla R3 (bullish) or below S3 (bearish) with volume confirmation and weekly trend filter. 
Weekly trend: price above/below 50-period EMA on weekly chart. 
Volume: current 6h volume > 1.5x average 6h volume over last 24 periods (4 days). 
Camarilla levels calculated from prior daily OHLC. 
Designed to work in both bull and bear markets by using weekly trend filter and volume confirmation to avoid false breakouts. 
Target: 15-35 trades/year.
"""

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly trend: EMA50 on weekly
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily OHLC for Camarilla levels (prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day: R3, S3
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_R3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_S3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe (use prior day's levels)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume confirmation: 6h volume > 1.5x average of last 24 periods (4 days)
    vol_ma_24 = np.full(n, np.nan)
    if n >= 24:
        vol_ma_24[23] = np.mean(volume[:24])
        for i in range(24, n):
            vol_ma_24[i] = (vol_ma_24[i-1] * 23 + volume[i]) / 24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # volume MA warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_ma_24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_confirm = volume[i] > 1.5 * vol_ma_24[i]
        
        if position == 0:
            # Long: Close above R3 with volume and weekly uptrend (price > EMA50 weekly)
            if close[i] > camarilla_R3_aligned[i] and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 with volume and weekly downtrend (price < EMA50 weekly)
            elif close[i] < camarilla_S3_aligned[i] and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S3 (reversal signal)
            if close[i] < camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 (reversal signal)
            if close[i] > camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals