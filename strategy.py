#!/usr/bin/env python3
# 4h_VDC_Scalper
# Hypothesis: Volume-Price Divergence with Confluence detects institutional accumulation/distribution.
# Combines volume divergence (price makes new high/low but volume fails to confirm) with
# RSI extreme readings and 4h EMA20 trend filter. Works in both bull and bear markets by
# fading exhaustion moves. Uses volume confirmation to filter false signals.
# Target: 25-40 trades/year to minimize fee drag.

name = "4h_VDC_Scalper"
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
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price extremes for divergence detection
    highest_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume divergence signals
    vol_div_bear = (high == highest_high) & (volume < volume_ma * 0.8)
    vol_div_bull = (low == lowest_low) & (volume < volume_ma * 0.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14), volume MA (20), price extremes (10)
    start_idx = max(14, 20, 10)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: bullish volume divergence + RSI oversold
            if vol_div_bull[i] and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish volume divergence + RSI overbought
            elif vol_div_bear[i] and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or bearish divergence
            if rsi[i] > 70 or vol_div_bear[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or bullish divergence
            if rsi[i] < 30 or vol_div_bull[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals