#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI2_Confirm
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) as primary trend filter on 4h, 
combined with RSI(2) for precise entry timing and volume confirmation for confirmation. 
KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing 
trends. RSI(2) provides oversold/overbought entries in trending markets. Volume 
filter ensures institutional participation. Works in both bull (trend following) and 
bear (mean reversion within trend) markets. Target: 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate KAMA (trend) on primary timeframe (4h)
    close = prices['close'].values
    direction = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(2) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    avg_loss = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price above KAMA (uptrend) + RSI(2) < 10 (oversold) + volume spike
            if (price_close > kama_val and 
                rsi_val < 10 and 
                vol_spike > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) + RSI(2) > 90 (overbought) + volume spike
            elif (price_close < kama_val and 
                  rsi_val > 90 and 
                  vol_spike > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses KAMA (trend change)
            if position == 1 and price_close < kama_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_KAMA_Trend_RSI2_Confirm"
timeframe = "4h"
leverage = 1.0