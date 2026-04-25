#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_RSI_Filter_v1
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 12h for trend direction with RSI(14) filter on 1d to avoid counter-trend entries. 
KAMA adapts to market noise, reducing whipsaw in choppy conditions. RSI filter ensures we only take trend-aligned entries when momentum is not extreme.
Volume confirmation on 12h adds validity to breakouts. Designed for fewer trades (12-37/year) to minimize fee drag on 12h timeframe.
Works in both bull and bear markets by following adaptive trend with momentum filter.
"""

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
    
    # Get 1d data for HTF RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI(14) for momentum filter
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # neutral when undefined
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate KAMA on 12h (ER=10, fast=2, slow=30)
    # Efficiency Ratio = |change| / sum(|deltas|)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Correct ER calculation per bar
    er = np.zeros(n)
    for i in range(10, n):  # need 10 bars for ER calculation
        if i >= 1:
            change_val = np.abs(close[i] - close[i-10])
            volatility_val = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if volatility_val > 0:
                er[i] = change_val / volatility_val
            else:
                er[i] = 1.0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: 12h volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10) and volume MA (20)
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI filter: avoid extreme momentum (overbought/oversold)
        rsi_not_extreme = (rsi_1d_aligned[i] > 20) and (rsi_1d_aligned[i] < 80)
        
        if position == 0:
            # Long setup: price above KAMA + RSI not extreme + volume spike
            long_setup = price_above_kama and rsi_not_extreme and volume_spike[i]
            
            # Short setup: price below KAMA + RSI not extreme + volume spike
            short_setup = price_below_kama and rsi_not_extreme and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR RSI becomes extremely overbought
            if (not price_above_kama) or (rsi_1d_aligned[i] >= 80):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR RSI becomes extremely oversold
            if (not price_below_kama) or (rsi_1d_aligned[i] <= 20):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_With_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0