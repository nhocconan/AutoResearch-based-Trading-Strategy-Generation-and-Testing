#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Trend_Filter
# Hypothesis: KAMA direction (trend) on 4h with RSI(14) filter for overbought/oversold conditions and volume confirmation.
# KAMA adapts to market noise, providing a smooth trend line that avoids whipsaws in choppy markets.
# RSI filter prevents entries in extreme conditions, improving win rate in both bull and bear markets.
# Volume confirmation ensures trades have conviction. Designed for minimal trades to avoid fee drag.

name = "4h_KAMA_Direction_RSI_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === KAMA (4h) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[:10] = 0  # First 10 values invalid
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Will be corrected below
    # Recompute volatility properly: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # Same length as close
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Average gain/loss over 14 periods
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI filter: avoid extremes
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: price above KAMA, RSI not overbought, volume confirmation
            if price_above_kama and rsi_not_overbought and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: price below KAMA, RSI not oversold, volume confirmation
            elif price_below_kama and rsi_not_oversold and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price falls below KAMA or RSI overbought
            if not price_above_kama or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above KAMA or RSI oversold
            if not price_below_kama or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals