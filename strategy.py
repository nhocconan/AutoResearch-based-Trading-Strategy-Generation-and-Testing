#!/usr/bin/env python3
"""
12h_RSI_Trend_Filter
Hypothesis: Uses RSI(14) with EMA(50) trend filter on 12h timeframe.
Enters long when RSI crosses above 50 with EMA50 rising, short when RSI crosses below 50 with EMA50 falling.
Designed for moderate trade frequency (~20-30/year) with trend-following capability in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # EMA(50) for trend filter
    ema50 = np.full(n, np.nan)
    k = 2 / (50 + 1)
    for i in range(50, n):
        if i == 50:
            ema50[i] = np.mean(close[0:51])
        else:
            ema50[i] = close[i] * k + ema50[i-1] * (1 - k)
    
    # EMA50 slope (trend direction)
    ema50_slope = np.full(n, np.nan)
    for i in range(51, n):
        ema50_slope[i] = ema50[i] - ema50[i-1]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema50[i]) or np.isnan(ema50_slope[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI crosses above 50 with rising EMA50 and volume spike
            if rsi[i] > 50 and rsi[i-1] <= 50 and ema50_slope[i] > 0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 50 with falling EMA50 and volume spike
            elif rsi[i] < 50 and rsi[i-1] >= 50 and ema50_slope[i] < 0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI crosses below 50 or EMA50 turns down
            if rsi[i] < 50 or ema50_slope[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI crosses above 50 or EMA50 turns up
            if rsi[i] > 50 or ema50_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_Trend_Filter"
timeframe = "12h"
leverage = 1.0