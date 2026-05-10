#!/usr/bin/env python3
"""
6h_AdaptiveKeltnerChannels
Hypothesis: Price breaks adaptive Keltner Channels (EMA20 +/- ATR*multiplier) with volume confirmation and trend alignment from 1d EMA50. Keltner Channels adapt to volatility, reducing false signals in low-volatility environments. Trend filter ensures alignment with higher timeframe direction, improving performance in both bull and bear markets. Volume confirmation filters out weak breakouts. Target: 15-30 trades/year.
"""

name = "6h_AdaptiveKeltnerChannels"
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
    
    # Keltner Channel parameters
    ema_period = 20
    atr_period = 14
    multiplier = 2.0
    
    # Calculate EMA20
    ema = np.full(n, np.nan)
    if n >= ema_period:
        ema[ema_period-1] = np.mean(close[:ema_period])
        alpha = 2 / (ema_period + 1)
        for i in range(ema_period, n):
            ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    
    # Calculate True Range and ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate Keltner Channels
    upper_keltner = ema + multiplier * atr
    lower_keltner = ema - multiplier * atr
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha_50 = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha_50 * close_1d[i] + (1 - alpha_50) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
    
    start_idx = max(ema_period, atr_period, 50)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema[i]) or np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        # Approximate 6h volume from 1d: 1d volume / 4 (since 24h/6h = 4)
        vol_6h_approx = vol_sma_20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.5 * vol_6h_approx
        
        if position == 0:
            # Long: Break above upper Keltner with uptrend and volume
            if close[i] > upper_keltner[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Keltner with downtrend and volume
            elif close[i] < lower_keltner[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA (trend reversal)
            if close[i] < ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA (trend reversal)
            if close[i] > ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals