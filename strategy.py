#!/usr/bin/env python3
"""
4h_1d_kama_rsi_volatility_breakout
Hypothesis: 4-hour KAMA trend direction combined with RSI momentum and volatility-adjusted breakouts.
Uses 1-day ATR for volatility filtering to adapt to changing market conditions.
Works in bull/bear by using adaptive trend (KAMA) and momentum (RSI) with volatility filters.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

name = "4h_1d_kama_rsi_volatility_breakout"
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
    
    # Get daily data for ATR and KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.subtract(close_1d, np.roll(close_1d, 10)))
    volatility = np.sum(np.abs(np.subtract(close_1d, np.roll(close_1d, 1))), axis=0) if False else None
    # Proper volatility calculation for ER
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    volatility = np.roll(volatility, 1)
    volatility[0] = 0
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI calculation (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR for volatility filter (14-day ATR)
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align KAMA, RSI, and ATR to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volatility-adjusted breakout levels
    # Upper band: KAMA + 1.5 * ATR
    # Lower band: KAMA - 1.5 * ATR
    upper_band = kama_aligned + 1.5 * atr_aligned
    lower_band = kama_aligned - 1.5 * atr_aligned
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price crosses above upper band with RSI > 50 and volume confirmation
        if (close[i] > upper_band[i] and rsi_aligned[i] > 50 and 
            vol_confirm[i] and atr_aligned[i] > 0 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price crosses below lower band with RSI < 50 and volume confirmation
        elif (close[i] < lower_band[i] and rsi_aligned[i] < 50 and 
              vol_confirm[i] and atr_aligned[i] > 0 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price crosses back to KAMA
        elif position == 1 and close[i] < kama_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > kama_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals