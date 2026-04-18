#!/usr/bin/env python3
"""
4h_RSI_Overbought_Oversold_With_Trend_Filter
Hypothesis: Uses RSI(14) for mean reversion entries with EMA(50) trend filter on 4h.
Enters long when RSI < 30 and price > EMA50 (bullish mean reversion in uptrend).
Enters short when RSI > 70 and price < EMA50 (bearish mean reversion in downtrend).
Requires volume > 1.5x 20-period average for confirmation.
Designed for fewer trades (~20-30/year) with high win rate in both bull and bear markets.
"""

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
    
    # RSI(14) calculation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[i-rsi_period+1:i+1])
            avg_loss[i] = np.mean(loss[i-rsi_period+1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA(50) trend filter
    ema_period = 50
    ema = np.full(n, np.nan)
    k = 2 / (ema_period + 1)
    for i in range(ema_period, n):
        if i == ema_period:
            ema[i] = np.mean(close[i-ema_period+1:i+1])
        else:
            ema[i] = close[i] * k + ema[i-1] * (1 - k)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold in uptrend with volume spike
            if rsi[i] < 30 and close[i] > ema[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in downtrend with volume spike
            elif rsi[i] > 70 and close[i] < ema[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral or trend changes
            if rsi[i] > 50 or close[i] < ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral or trend changes
            if rsi[i] < 50 or close[i] > ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Overbought_Oversold_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0