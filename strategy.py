#!/usr/bin/env python3
"""
1h_Keltner_Channel_Breakout_4hTrend_1dVolume
Hypothesis: 1h price breaks Keltner Channel (20, 2.0) in direction of 4h EMA50 trend with 1d volume spike confirmation.
Keltner Channels adapt to volatility via ATR, providing dynamic support/resistance.
EMA50 trend filter ensures alignment with medium-term direction, working in both bull and bear markets.
Volume confirmation on 1d reduces false breakouts. Target: 15-35 trades/year.
"""

name = "1h_Keltner_Channel_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_50_4h[i-1]
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume SMA for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma_20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma_20_1d[i] = (vol_sma_20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # Keltner Channel (20, 2.0) on 1h
    kc_period = 20
    kc_mult = 2.0
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # ATR
    atr = np.full(n, np.nan)
    for i in range(kc_period, n):
        atr[i] = np.mean(tr[i-kc_period:i])
    
    # EMA of close for Keltner middle line
    ema_close = np.full(n, np.nan)
    if n >= kc_period:
        ema_close[kc_period-1] = np.mean(close[:kc_period])
        alpha = 2 / (kc_period + 1)
        for i in range(kc_period, n):
            ema_close[i] = alpha * close[i] + (1 - alpha) * ema_close[i-1]
    
    # Keltner Bands
    upper_keltner = ema_close + kc_mult * atr
    lower_keltner = ema_close - kc_mult * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kc_period, 20, 50)  # Keltner + volume + trend warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_close[i]) or np.isnan(atr[i]) or np.isnan(vol_sma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average 1d volume (scaled)
        # Approximate 1h volume from 1d: 1d volume / 24 (assuming uniform distribution)
        vol_1h_approx = vol_sma_20_1d_aligned[i] / 24.0
        volume_confirm = volume[i] > 1.5 * vol_1h_approx
        
        if position == 0:
            # Long: Break above upper Keltner and above 4h EMA50
            if close[i] > upper_keltner[i] and close[i] > ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: Break below lower Keltner and below 4h EMA50
            elif close[i] < lower_keltner[i] and close[i] < ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Close below EMA (middle line)
            if close[i] < ema_close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Close above EMA (middle line)
            if close[i] > ema_close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals