#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from 1d with breakout at R3/S3 in direction of 1d EMA34 trend, confirmed by volume spikes.
Targets 15-25 trades/year per symbol with strict entry conditions to minimize fee drag.
Works in both bull and bear markets by following 1d trend and using volatility-based entries.
"""

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla formula:
    # H4 = C + (H-L) * 1.1/2
    # L4 = C - (H-L) * 1.1/2
    # H3 = C + (H-L) * 1.1/4
    # L3 = C - (H-L) * 1.1/4
    # H2 = C + (H-L) * 1.1/6
    # L2 = C - (H-L) * 1.1/6
    # H1 = C + (H-L) * 1.1/12
    # L1 = C - (H-L) * 1.1/12
    # We only need R3/S3 (H3/L3) and R4/S4 (H4/L4)
    
    # For each 6h bar, we need the previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_H3 = np.full(len(close_1d), np.nan)
    camarilla_L3 = np.full(len(close_1d), np.nan)
    camarilla_H4 = np.full(len(close_1d), np.nan)
    camarilla_L4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        range_ = H - L
        
        camarilla_H3[i] = C + range_ * 1.1 / 4
        camarilla_L3[i] = C - range_ * 1.1 / 4
        camarilla_H4[i] = C + range_ * 1.1 / 2
        camarilla_L4[i] = C - range_ * 1.1 / 2
    
    # Align Camarilla levels to 6b timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Calculate volume SMA(20) for volume confirmation
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 35)
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Long: Break above R3 with uptrend and volume confirmation
            if close[i] > H3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with downtrend and volume confirmation
            elif close[i] < L3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses back below EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses back above EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals