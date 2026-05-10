#!/usr/bin/env python3
"""
4H_RSI_Divergence_With_Volume_Confirmation
Hypothesis: Combines RSI divergence (bullish/bearish) with volume confirmation and 4h EMA50 trend filter to capture high-probability reversals. Designed for low trade frequency (<25/year) to minimize fee burn while maintaining edge in both bull and bear markets by trading against overextended moves in the direction of the higher timeframe trend.
"""

name = "4H_RSI_Divergence_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # RSI divergence detection (lookback 5 bars)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(5, n):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if low[i] < low[i-5] and rsi[i] > rsi[i-5]:
            # Confirm with higher low in price within lookback
            if low[i] >= min(low[i-4:i+1]):
                bullish_div[i] = True
        # Bearish divergence: price makes higher high, RSI makes lower high
        if high[i] > high[i-5] and rsi[i] < rsi[i-5]:
            # Confirm with lower high in price within lookback
            if high[i] <= max(high[i-4:i+1]):
                bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50[i]) or np.isnan(rsi[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend
        is_uptrend = close[i] > ema_50[i]
        is_downtrend = close[i] < ema_50[i]
        
        if position == 0:
            # Long entry: bullish divergence + volume confirmation + uptrend
            if bullish_div[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish divergence + volume confirmation + downtrend
            elif bearish_div[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish divergence or trend break
            if bearish_div[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish divergence or trend break
            if bullish_div[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals