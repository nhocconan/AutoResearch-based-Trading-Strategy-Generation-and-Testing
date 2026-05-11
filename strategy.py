#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Filter_v1
Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) on 12h to determine trend direction,
filtered by RSI on 1d for overbought/oversold conditions. In bull/bear markets, we follow
the KAMA trend when RSI is not extreme. In ranging markets (when RSI is between 40-60),
we avoid trades to prevent whipsaw. Designed for low trade frequency (~15-25 trades/year)
by requiring alignment between 12h trend and 1d RSI conditions.
"""

name = "12h_KAMA_Direction_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    
    # --- KAMA (10-period ER, 2/30 smoothing) on 12h close ---
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period sum of absolute changes
    # Vectorized volatility calculation
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+10]))) if i+10 <= len(close) else 0 for i in range(len(close))])
    volatility[:9] = 0  # Pad beginning
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # (ER*(fastest-slowest) + slowest)^2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- RSI (14-period) on 1d close ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = np.full_like(close_1d, np.nan, dtype=float)
    avg_loss = np.full_like(close_1d, np.nan, dtype=float)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)  # Avoid division by zero
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.where(np.isnan(rsi_1d), 50, rsi_1d)  # Neutral when undefined
    
    # Align RSI to 12h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend from KAMA
        bullish = close[i] > kama[i]
        bearish = close[i] < kama[i]
        
        # RSI conditions
        rsi_overbought = rsi_1d_aligned[i] > 60
        rsi_oversold = rsi_1d_aligned[i] < 40
        rsi_neutral = (rsi_1d_aligned[i] >= 40) & (rsi_1d_aligned[i] <= 60)
        
        if position == 0:
            # Enter long: bullish KAMA and not overbought RSI
            if bullish and not rsi_overbought:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish KAMA and not oversold RSI
            elif bearish and not rsi_oversold:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: bearish KAMA or overbought RSI
                if bearish or rsi_overbought:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish KAMA or oversold RSI
                if bullish or rsi_oversold:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals