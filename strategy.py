#!/usr/bin/env python3
"""
4h_200MA_50MA_200Bias_Trend_v1
Concept: 4h EMA(200) trend filter with SMA(50) momentum and 200-period price bias.
- Long: Price > EMA200 AND SMA50 > SMA50[5] AND close > open[200] (bullish bias)
- Short: Price < EMA200 AND SMA50 < SMA50[5] AND close < open[200] (bearish bias)
- Exit: Price crosses EMA200
- Position sizing: 0.30
- Target: 15-40 trades/year (60-160 total over 4 years)
- Works in bull/bear: EMA200 defines trend, SMA50 captures momentum, 200-bar bias filters counter-trend noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_200MA_50MA_200Bias_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for 200-bar bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 4h: EMA200 trend filter ===
    close = prices['close'].values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # === 4h: SMA50 momentum ===
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # === Daily: 200-period price bias (bullish if close > open 200 periods ago) ===
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    # Calculate 200-period price change: (current close - open 200 periods ago) / open 200 periods ago
    price_change_200 = (close_1d - np.roll(open_1d, 200)) / np.roll(open_1d, 200)
    # Handle first 200 values
    price_change_200[:200] = np.nan
    price_change_200_aligned = align_htf_to_ltf(prices, df_1d, price_change_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema200_val = ema200[i]
        sma50_val = sma50[i]
        sma50_prev = sma50[i-5] if i >= 5 else np.nan  # 5-period momentum
        bias_val = price_change_200_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema200_val) or np.isnan(sma50_val) or np.isnan(sma50_prev) or 
            np.isnan(bias_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above EMA200 AND SMA50 rising AND bullish 200-bar bias
            if close[i] > ema200_val and sma50_val > sma50_prev and bias_val > 0:
                signals[i] = 0.30
                position = 1
            # Short: Price below EMA200 AND SMA50 falling AND bearish 200-bar bias
            elif close[i] < ema200_val and sma50_val < sma50_prev and bias_val < 0:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below EMA200
            if close[i] < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: Price crosses above EMA200
            if close[i] > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals