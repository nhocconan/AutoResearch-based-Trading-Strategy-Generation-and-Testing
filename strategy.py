#!/usr/bin/env python3
"""
12h_KAMA_Direction_With_RSI_And_Volume_Spike
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to capture adaptive trend direction on 12h timeframe, confirmed by RSI momentum and volume spikes. KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends. Volume spike confirms institutional interest. Designed for 15-30 trades/year to work in both bull and bear markets via adaptive trend filter.
"""

name = "12h_KAMA_Direction_With_RSI_And_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (ER=10, fast=2, slow=30) on 12h close
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Fix: volatility should be rolling sum of absolute changes
    volatility = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=1).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constant
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to ensure no look-ahead (already on 12h, but ensure proper start)
    kama_aligned = kama  # Already on same timeframe
    
    # Calculate daily RSI(14)
    daily_close = df_1d['close'].values
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi, additional_delay_bars=0)
    
    # Calculate 12h volume spike (ratio to 20-period MA)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for RSI and KAMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine KAMA trend
        kama_trend_up = close[i] > kama_aligned[i]
        kama_trend_down = close[i] < kama_aligned[i]
        
        if position == 0:
            # Long: price above KAMA, RSI > 50 (bullish momentum), volume spike
            if (kama_trend_up and 
                rsi_aligned[i] > 50 and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50 (bearish momentum), volume spike
            elif (kama_trend_down and 
                  rsi_aligned[i] < 50 and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI turns bearish
            if not kama_trend_up or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI turns bullish
            if not kama_trend_down or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals