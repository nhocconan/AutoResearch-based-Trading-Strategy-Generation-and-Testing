#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI20_CloseOnly
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 12h to capture trend direction, with RSI(2) for mean-reversion entries on pullbacks, and volume confirmation. Designed to work in both bull and bear markets by following the trend on higher timeframe while using RSI2 for entry timing. Target 15-30 trades per year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === KAMA on 12h (trend direction) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility properly
    volatility = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        volatility[i] = volatility[i-1] + np.abs(close_12h[i] - close_12h[i-1])
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # === RSI(2) on 12h ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(kama_12h_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_1d = ema_34_1d_aligned[i]
        trend_12h = kama_12h_aligned[i]
        rsi_val = rsi[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: RSI2 < 10 + price above both 1d and 12h trend + volume spike > 1.5
            if (rsi_val < 10 and 
                price_close > trend_1d and 
                price_close > trend_12h and
                vol_spike > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: RSI2 > 90 + price below both 1d and 12h trend + volume spike > 1.5
            elif (rsi_val > 90 and 
                  price_close < trend_1d and 
                  price_close < trend_12h and
                  vol_spike > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when RSI2 crosses 50 in opposite direction
            if position == 1 and rsi_val < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_val > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_Trend_RSI20_CloseOnly"
timeframe = "12h"
leverage = 1.0