#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Extreme_Volume
Hypothesis: 12-hour KAMA trend filter with RSI extremes and volume confirmation on 12-hour timeframe.
KAMA adapts to market efficiency, reducing whipsaw in ranging markets while capturing trends.
RSI extremes (overbought/oversold) provide mean-reversion entries aligned with the trend.
Volume confirms conviction. Designed for low frequency (12-37 trades/year) with robustness across bull/bear regimes.
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
    
    # Get 12h data for indicators
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate KAMA (10, 2, 30) on 12h
    close_12h = df_12h['close'].values
    kama_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 30:
        # Efficiency ratio
        change = np.abs(np.diff(close_12h, 10))
        volatility = np.sum(np.abs(np.diff(close_12h, 1)), axis=1)
        er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        # Initialize KAMA
        kama_12h[29] = np.mean(close_12h[0:30])
        for i in range(30, len(close_12h)):
            kama_12h[i] = kama_12h[i-1] + sc[i-30] * (close_12h[i] - kama_12h[i-1])
    
    # Calculate RSI (14) on 12h
    rsi_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 15:
        delta = np.diff(close_12h)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(len(close_12h), np.nan)
        avg_loss = np.full(len(close_12h), np.nan)
        avg_gain[14] = np.mean(gain[0:14])
        avg_loss[14] = np.mean(loss[0:14])
        for i in range(15, len(close_12h)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi_12h = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 2.0 x 20-period average on 12h
    vol_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(vol_12h), np.nan)
    for i in range(20, len(vol_12h)):
        vol_ma_12h[i] = np.mean(vol_12h[i-20:i])
    vol_spike_12h = vol_12h > (vol_ma_12h * 2.0)
    
    # Align 12h indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) with volume spike and price above KAMA (uptrend)
            if (rsi_aligned[i] < 30 and vol_spike_aligned[i] and 
                close[i] > kama_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) with volume spike and price below KAMA (downtrend)
            elif (rsi_aligned[i] > 70 and vol_spike_aligned[i] and 
                  close[i] < kama_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 70 (overbought) or price below KAMA (trend change)
            if (rsi_aligned[i] > 70 or close[i] < kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 30 (oversold) or price above KAMA (trend change)
            if (rsi_aligned[i] < 30 or close[i] > kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_RSI_Extreme_Volume"
timeframe = "12h"
leverage = 1.0