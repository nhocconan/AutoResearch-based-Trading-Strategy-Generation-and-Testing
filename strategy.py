#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Price breaks Camarilla R1 or S1 level calculated from prior 1d session, with confirmation from 1d EMA34 trend and volume spike. Camarilla levels provide high-probability reversal points in ranging markets while acting as breakout levels in trending markets. EMA34 trend filter ensures alignment with higher timeframe direction. Volume confirmation reduces false breakouts. Target: 20-50 trades/year (80-200 total over 4 years).
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # 1d data for Camarilla levels and filters
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each 1d bar (based on prior day)
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    camarilla_R1_1d = np.full(len(close_1d), np.nan)
    camarilla_S1_1d = np.full(len(close_1d), np.nan)
    
    if len(high_1d) >= 1:
        for i in range(1, len(high_1d)):
            h_l = high_1d[i-1] - low_1d[i-1]
            camarilla_R1_1d[i] = close_1d[i-1] + h_l * 1.1 / 12
            camarilla_S1_1d[i] = close_1d[i-1] - h_l * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_R1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume SMA20 for volume confirmation
    vol_sma_20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma_20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma_20_1d[i] = (vol_sma_20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or \
           np.isnan(camarilla_R1_1d_aligned[i]) or np.isnan(camarilla_S1_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        # Approximate 4h volume from 1d: 1d volume / 6 (since 24h/4h = 6)
        vol_4h_approx = vol_sma_20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: Break above Camarilla R1 with uptrend and volume
            if close[i] > camarilla_R1_1d_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 with downtrend and volume
            elif close[i] < camarilla_S1_1d_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA34 (trend reversal)
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA34 (trend reversal)
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals