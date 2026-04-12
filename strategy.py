#!/usr/bin/env python3
"""
12h_1d_rsi_divergence_v1
Uses daily RSI for trend bias and 12h RSI divergence for entry.
Long when daily RSI > 50 and 12h shows bullish RSI divergence (price lower low, RSI higher low).
Short when daily RSI < 50 and 12h shows bearish RSI divergence (price higher high, RSI lower high).
Volume confirmation required. Designed for low trade frequency (~20-30 trades/year).
Works in bull markets via trend-following and in bear markets via mean-reversion divergences.
"""

name = "12h_1d_rsi_divergence_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(series, period=14):
    """Relative Strength Index"""
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(series)
    avg_loss = np.zeros_like(series)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(series)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_val = 100 - (100 / (1 + rs))
    
    # Pad beginning
    rsi_full = np.full_like(series, np.nan)
    rsi_full[period:] = rsi_val[period:]
    return rsi_full

def find_divergence(price, rsi_val, lookback=5):
    """
    Find bullish/bearish divergence
    Returns: 1 for bullish divergence, -1 for bearish divergence, 0 otherwise
    """
    if len(price) < lookback * 2:
        return 0
    
    # Look for recent swing low/high
    bullish_div = 0
    bearish_div = 0
    
    # Check for bullish divergence: price makes lower low, RSI makes higher low
    for i in range(lookback, len(price)-lookback):
        # Find local minimum in price
        if price[i] == np.min(price[i-lookback:i+lookback+1]):
            # Check if this is lower than previous low
            prev_low_idx = np.argmin(price[i-2*lookback:i-lookback+1]) + i-2*lookback
            if price[i] < price[prev_low_idx] and rsi_val[i] > rsi_val[prev_low_idx]:
                bullish_div = 1
                break
    
    # Check for bearish divergence: price makes higher high, RSI makes lower high
    for i in range(lookback, len(price)-lookback):
        # Find local maximum in price
        if price[i] == np.max(price[i-lookback:i+lookback+1]):
            # Check if this is higher than previous high
            prev_high_idx = np.argmax(price[i-2*lookback:i-lookback+1]) + i-2*lookback
            if price[i] > price[prev_high_idx] and rsi_val[i] < rsi_val[prev_high_idx]:
                bearish_div = -1
                break
    
    return bullish_div if bullish_div != 0 else bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily RSI for trend bias (>50 = bullish bias, <50 = bearish bias)
    rsi_1d = rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 12h RSI for divergence detection
    rsi_12h = rsi(close, 14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_12h[i]):
            signals[i] = 0.0
            continue
        
        # Check for divergence on recent window
        lookback = 10
        if i >= lookback:
            div_signal = find_divergence(close[i-lookback:i+1], rsi_12h[i-lookback:i+1], lookback//2)
        else:
            div_signal = 0
        
        # Long entry: daily RSI bullish (>50) + bullish divergence + volume
        if (rsi_1d_aligned[i] > 50 and div_signal == 1 and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: daily RSI bearish (<50) + bearish divergence + volume
        elif (rsi_1d_aligned[i] < 50 and div_signal == -1 and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: RSI crosses back to neutral zone (40-60) or opposite divergence
        elif position == 1 and (rsi_1d_aligned[i] < 40 or div_signal == -1):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_1d_aligned[i] > 60 or div_signal == 1):
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