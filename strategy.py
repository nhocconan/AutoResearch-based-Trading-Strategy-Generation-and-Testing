#!/usr/bin/env python3

"""
4h_1d_kama_rsi_volatility_breakout
Hypothesis: 4-hour KAMA trend with RSI momentum and volatility breakout. KAMA adapts to market noise,
reducing false signals in chop. RSI filters momentum extremes. Volatility breakout captures
expansion after contraction. Works in bull/bear by adapting trend strength and using volatility
filters to avoid false breakouts. Target: 20-50 trades/year (80-200 total over 4 years).
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
    
    # Get daily data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    # Efficiency Ratio
    change = np.abs(np.subtract(close_1d, np.roll(close_1d, 10)))
    volatility = np.sum(np.abs(np.subtract(close_1d, np.roll(close_1d, 1))), axis=0)  # will fix below
    
    # Proper volatility calculation (sum of absolute changes over 10 periods)
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.subtract(close_1d[i-9:i+1], np.roll(close_1d[i-9:i+1], 1))))
    
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)
    # KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # start after first 10 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI calculation (14-period)
    delta = np.subtract(close_1d, np.roll(close_1d, 1))
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR for volatility breakout (10-day ATR)
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volatility breakout: price > KAMA + 0.5 * ATR (long) or < KAMA - 0.5 * ATR (short)
    vol_threshold = 0.5 * atr_aligned
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price above KAMA + volatility threshold, RSI > 50, volume confirmation
        if (close[i] > kama_aligned[i] + vol_threshold[i] and 
            rsi_aligned[i] > 50 and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below KAMA - volatility threshold, RSI < 50, volume confirmation
        elif (close[i] < kama_aligned[i] - vol_threshold[i] and 
              rsi_aligned[i] < 50 and vol_confirm[i] and position != -1):
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