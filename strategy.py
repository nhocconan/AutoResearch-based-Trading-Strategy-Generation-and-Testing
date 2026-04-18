#!/usr/bin/env python3
"""
4h_RSI2_Pullback_TrendFollow
Hypothesis: RSI(2) pullbacks in strong trends (EMA50) with volume confirmation.
Works in bull markets by buying dips and in bear markets by selling rallies.
Low frequency due to strict RSI(2)<10/ >90 and EMA50 filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(2) - faster for pullback signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    for i in range(n):
        if i < 2:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] + gain[i]) / 2
                avg_loss[i] = (avg_loss[i-1] + loss[i]) / 2
        else:
            avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
            avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA50 trend filter
    ema50 = np.full(n, np.nan)
    if n >= 50:
        ema50[49] = np.mean(close[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, n):
            ema50[i] = close[i] * alpha + ema50[i-1] * (1 - alpha)
    
    # Volume filter: current > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema50[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) + price > EMA50 (uptrend) + volume
            if rsi[i] < 10 and close[i] > ema50[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (overbought) + price < EMA50 (downtrend) + volume
            elif rsi[i] > 90 and close[i] < ema50[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI(2) > 50 (mean reversion) or trend change
            if rsi[i] > 50 or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI(2) < 50 (mean reversion) or trend change
            if rsi[i] < 50 or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI2_Pullback_TrendFollow"
timeframe = "4h"
leverage = 1.0