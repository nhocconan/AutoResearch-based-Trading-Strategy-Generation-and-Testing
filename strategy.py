#!/usr/bin/env python3
"""
4h_TrendFollowing_RSI4060
Hypothesis: Trend following with RSI filter on 4h timeframe. Uses 4h EMA20 for trend direction and RSI(14) for momentum confirmation.
Enters long when price > EMA20 and RSI between 40-60 in uptrend, short when price < EMA20 and RSI between 40-60 in downtrend.
Adds volume confirmation (current volume > 1.5x 20-period volume average) to filter false signals.
Designed to work in both bull and bear markets by following the trend while avoiding overextended conditions.
Targets 20-50 trades per year to minimize fee drag.
"""

name = "4h_TrendFollowing_RSI4060"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA20 for trend
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i-1]
    
    # RSI(14)
    rsi = np.full(n, np.nan)
    if n >= 15:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[14] = np.mean(gain[:14])
        avg_loss[14] = np.mean(loss[:14])
        
        for i in range(15, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:14] = np.nan
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg20 = np.full(n, np.nan)
    if n >= 20:
        vol_avg20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_avg20[i] = (vol_avg20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 15)  # Need EMA20 and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema20[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_avg20[i]
        
        if position == 0:
            # Long: Uptrend (price > EMA20) with RSI in neutral range and volume confirmation
            if close[i] > ema20[i] and 40 <= rsi[i] <= 60 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < EMA20) with RSI in neutral range and volume confirmation
            elif close[i] < ema20[i] and 40 <= rsi[i] <= 60 and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Trend reversal or RSI overbought
            if close[i] < ema20[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Trend reversal or RSI oversold
            if close[i] > ema20[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals