#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Filter_Volume
# Hypothesis: KAMA adapts to market efficiency, capturing trend while avoiding whipsaws in sideways markets.
# Combined with RSI(14) for momentum confirmation and volume filter, this strategy aims to capture
# strong moves in both bull and bear markets with low frequency (target: 20-40 trades/year).
# KAMA crosses above/below price signal trend changes; RSI filters for strength; volume confirms conviction.

name = "4h_KAMA_Trend_RSI_Filter_Volume"
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
    
    # === KAMA ( Kaufman Adaptive Moving Average ) ===
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will fix below
    # Recompute volatility properly: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(len(close)):
        if i < 10:
            volatility[i] = np.nan
        else:
            volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # first 14 gains
    avg_loss[13] = np.mean(loss[1:14])  # first 14 losses
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume filter (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI conditions: avoid extremes, look for momentum
        rsi_bull = 50 < rsi[i] < 70   # bullish momentum
        rsi_bear = 30 < rsi[i] < 50   # bearish momentum
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price above KAMA, bullish RSI momentum, volume confirmation
            if price_above_kama and rsi_bull and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, bearish RSI momentum, volume confirmation
            elif price_below_kama and rsi_bear and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or RSI turns bearish
            if price_below_kama or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or RSI turns bullish
            if price_above_kama or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals