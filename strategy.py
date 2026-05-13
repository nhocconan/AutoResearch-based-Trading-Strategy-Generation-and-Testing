#!/usr/bin/env python3
"""
4H_RSI_Momentum_With_Volume_Filter
Hypothesis: RSI momentum combined with volume confirmation captures trend continuations in both bull and bear markets. Uses RSI > 55 for long momentum and RSI < 45 for short momentum, requiring volume > 1.5x average to confirm institutional participation. Includes 4h trend filter via EMA(50) to avoid counter-trend trades.
"""

name = "4H_RSI_Momentum_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA(50) for trend filter
    ema_period = 50
    ema = np.zeros_like(close)
    ema[0] = close[0]
    alpha = 2 / (ema_period + 1)
    for i in range(1, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: RSI > 55 (bullish momentum) + price > EMA(50) (uptrend) + volume confirmation
            if rsi[i] > 55 and close[i] > ema[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 45 (bearish momentum) + price < EMA(50) (downtrend) + volume confirmation
            elif rsi[i] < 45 and close[i] < ema[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 50 (loss of momentum) or price < EMA(50) (trend break)
            if rsi[i] < 50 or close[i] < ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 50 (loss of bearish momentum) or price > EMA(50) (trend break)
            if rsi[i] > 50 or close[i] > ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals