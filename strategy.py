#!/usr/bin/env python3
# 4H_KAMA_Direction_1dRSI_Filter_Trend
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) captures trend direction with low whipsaw, filtered by daily RSI to avoid counter-trend trades.
# In bull markets: KAMA rising + RSI > 50 = long; in bear markets: KAMA falling + RSI < 50 = short.
# Uses volume confirmation to ensure momentum and reduce false signals.
# Designed for 4h timeframe with discrete position sizing (0.25) to minimize fee churn.
# Target: 20-40 trades/year, effective in both bull and bear regimes by following adaptive trend.

name = "4H_KAMA_Direction_1dRSI_Filter_Trend"
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
    volume = prices['volume'].values
    
    # Get 1d data for RSI filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Calculate KAMA(10) on 4h close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    # Recompute volatility properly: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Warmup for KAMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction: price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            # Long entry: price above KAMA + RSI > 50 + volume spike
            if (price_above_kama and 
                rsi_1d_aligned[i] > 50 and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA + RSI < 50 + volume spike
            elif (price_below_kama and 
                  rsi_1d_aligned[i] < 50 and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI < 40
            if (price_below_kama or rsi_1d_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI > 60
            if (price_above_kama or rsi_1d_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals