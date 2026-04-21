# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI20_CloseOnly
Hypothesis: Use 4h KAMA (ER=10) as primary trend filter and RSI(2) for mean-reversion entries in the direction of the trend. Entry when RSI(2) < 10 (long) or > 90 (short) with price close crossing KAMA. Exit when RSI(2) crosses 50 in the opposite direction. Uses volume confirmation (volume > 1.3x 20-period average) to avoid false signals. Designed to work in both bull and bear markets by following adaptive trend while capturing exhaustion moves. Target 20-30 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # === 4h KAMA (ER=10) - adaptive trend filter ===
    close = prices['close'].values
    # Efficiency Ratio: |price change over 10 periods| / sum of absolute changes
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # This needs fixing - will use loop approach instead
    # Simpler: use pandas for ER calculation
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = abs(close_series.diff()).rolling(10).sum()
    er = change / volatility
    er = er.fillna(0).replace([np.inf, -np.inf], 0).values
    # Smoothing constants: SC = [ER*(fastest- slowest) + slowest]^2
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(2) for mean-reversion entries ===
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
    
    for i in range(30, n):  # Start after warmup period
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: RSI2 < 10 + price close crosses above KAMA + volume spike
            if (rsi_val < 10 and 
                price_close > kama_val and 
                close[i-1] <= kama[i-1] and
                vol_spike > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: RSI2 > 90 + price close crosses below KAMA + volume spike
            elif (rsi_val > 90 and 
                  price_close < kama_val and 
                  close[i-1] >= kama[i-1] and
                  vol_spike > 1.3):
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

name = "4h_KAMA_Trend_RSI20_CloseOnly"
timeframe = "4h"
leverage = 1.0