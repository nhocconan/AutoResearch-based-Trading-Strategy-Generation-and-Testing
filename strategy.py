#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_RSI_Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) on 12h captures trend strength while filtering noise, and RSI(14) on 12h provides overbought/oversold signals. Combining KAMA direction with RSI extremes creates a mean-reversion-with-trend strategy that works in both bull and bear markets. The 1-day trend filter (EMA50) ensures we trade with the higher timeframe trend, reducing false signals. Volume confirmation (>1.5x average) adds confirmation of participation. Target: 15-30 trades/year on 12h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for KAMA and RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h KAMA (ER=10, slow=2, fast=30)
    close_12h = df_12h['close'].values
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    direction = np.abs(np.subtract(close_12h, np.roll(close_12h, 10)))
    volatility = np.cumsum(change)
    volatility = np.where(volatility == 0, 1, volatility)
    er = direction / volatility
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Calculate 12h RSI(14)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros_like(close_1d)
    ema50_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_ok = volume_filter[i]
        
        # Mean reversion logic: 
        # Long when price < KAMA (below trend) AND RSI oversold (<30) AND uptrend filter (price > 1d EMA50)
        # Short when price > KAMA (above trend) AND RSI overbought (>70) AND downtrend filter (price < 1d EMA50)
        if position == 0:
            if price < kama_val and rsi_val < 30 and vol_ok and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif price > kama_val and rsi_val > 70 and vol_ok and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: price crosses back above KAMA (trend resumption) OR RSI overbought
            if price > kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back below KAMA (trend resumption) OR RSI oversold
            if price < kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
leverage = 1.0