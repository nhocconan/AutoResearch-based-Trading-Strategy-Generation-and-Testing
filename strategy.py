#!/usr/bin/env python3
"""
4h_Supertrend_RSI_Momentum
Hypothesis: Supertrend (ATR=10, multiplier=3) defines trend direction. RSI(14) measures momentum strength. Entry occurs when Supertrend confirms trend and RSI shows strong momentum (RSI>55 for long, RSI<45 for short) with volume confirmation. Exit when Supertrend reverses or momentum weakens. Designed for low trade frequency (target 20-40/year) to minimize fee drag in 4-hour bars.
"""

name = "4h_Supertrend_RSI_Momentum"
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
    
    # ATR calculation for Supertrend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Neutral RSI for warmup period
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if position == 0:
            # LONG: Supertrend uptrend, RSI > 55 (bullish momentum), volume confirmation
            if (direction[i] == 1 and 
                rsi_values[i] > 55 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Supertrend downtrend, RSI < 45 (bearish momentum), volume confirmation
            elif (direction[i] == -1 and 
                  rsi_values[i] < 45 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Supertrend reverses OR RSI drops below 50 (momentum loss)
            if (direction[i] == -1) or (rsi_values[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Supertrend reverses OR RSI rises above 50 (momentum loss)
            if (direction[i] == 1) or (rsi_values[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals