#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Price breaks Camarilla R1 (long) or S1 (short) levels calculated from prior 1d session, with confirmation from 1w EMA50 trend and volume spike. Uses 6h as primary timeframe for better trade frequency control. Camarilla levels provide high-probability reversal/breakout points, while weekly EMA50 ensures alignment with longer-term trend. Volume confirmation reduces false breakouts. Target: 15-30 trades/year.
"""

name = "6h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # 1d data for Camarilla levels (using prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar: R1, S1
    # R1 = close + (high - low) * 1.12
    # S1 = close - (high - low) * 1.12
    camarilla_r1_1d = close_1d + (high_1d - low_1d) * 1.12
    camarilla_s1_1d = close_1d - (high_1d - low_1d) * 1.12
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    camarilla_r1_6h = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_6h = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma_20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma_20_1d[i] = (vol_sma_20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or \
           np.isnan(camarilla_r1_6h[i]) or np.isnan(camarilla_s1_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled to 6h)
        # Approximate 6h volume from 1d: 1d volume / 4 (since 24h/6h = 4)
        vol_6h_approx = vol_sma_20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.5 * vol_6h_approx
        
        if position == 0:
            # Long: Break above Camarilla R1 with uptrend and volume
            if close[i] > camarilla_r1_6h[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 with downtrend and volume
            elif close[i] < camarilla_s1_6h[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA50 (trend reversal)
            if close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA50 (trend reversal)
            if close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals