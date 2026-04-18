#!/usr/bin/env python3
"""
1h_RSI_Trend_Pullback_WithVolume
Hypothesis: During strong 4h trends, RSI pullbacks on 1h with volume confirmation provide high-probability entries.
Uses 4h EMA50 for trend direction, 1h RSI(14) for pullback entries, and volume spike for confirmation.
Designed for low trade frequency (target: 15-30/year) with proper risk control via trend alignment.
Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).
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
    
    # 4h EMA50 for trend direction (calculated once)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 with proper smoothing
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = close_4h[i] * alpha + ema50_4h[i-1] * (1 - alpha)
    
    # Align 4h EMA50 to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h RSI(14) for pullback entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Wilder's smoothing
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full(n, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend + RSI pullback from oversold + volume spike
            if (close[i] > ema50_4h_aligned[i] and 
                rsi[i] < 35 and rsi[i] > 25 and  # Pullback from oversold
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + RSI pullback from overbought + volume spike
            elif (close[i] < ema50_4h_aligned[i] and 
                  rsi[i] > 65 and rsi[i] < 75 and  # Pullback from overbought
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h trend breaks down or RSI overbought
            if (close[i] < ema50_4h_aligned[i] or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h trend breaks up or RSI oversold
            if (close[i] > ema50_4h_aligned[i] or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Trend_Pullback_WithVolume"
timeframe = "1h"
leverage = 1.0