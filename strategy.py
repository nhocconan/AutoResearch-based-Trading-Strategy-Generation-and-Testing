#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing smooth trend direction.
In trending markets, KAMA follows price with low lag; in ranging markets, it flattens, reducing false signals.
Combined with RSI(14) for overbought/oversold conditions and volume confirmation, this filters entries to high-probability breakouts.
Designed for 20-30 trades/year, works in bull/bear via adaptive trend filter.
"""

name = "4h_KAMA_Direction_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for KAMA calculation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constants
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation: |close[i] - close[i-10]| / sum(|close[j] - close[j-1]|) for j=i-9 to i
    lookback = 10
    er = np.zeros(n)
    for i in range(lookback, n):
        price_change = np.abs(close[i] - close[i-lookback])
        price_volatility = np.sum(np.abs(np.diff(close[i-lookback:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (needs ~10 bars), RSI (14), volume EMA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        if position == 0:
            # Long: price above KAMA (uptrend) AND RSI < 30 (oversold) AND volume confirmation
            if close[i] > kama[i] and rsi[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) AND RSI > 70 (overbought) AND volume confirmation
            elif close[i] < kama[i] and rsi[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA OR RSI > 70 (overbought)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA OR RSI < 30 (oversold)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals