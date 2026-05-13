#!/usr/bin/env python3
"""
4h_RSI_Divergence_With_Volume_Filter
Hypothesis: RSI divergences signal exhaustion in momentum. Combined with volume exhaustion
and 4h trend filter, this captures reversals in both bull and bear markets. 
RSI bearish divergence: price makes higher high, RSI makes lower high → short signal.
RSI bullish divergence: price makes lower low, RSI makes higher low → long signal.
Volume filter ensures momentum is weakening. Position size 0.25 targets ~20-30 trades/year.
"""

name = "4h_RSI_Divergence_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA50
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume exhaustion: current volume < 0.7x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_exhaustion = volume < (0.7 * vol_ma)
    
    # Find swing points for divergence detection
    def find_swing_points(arr, window=3):
        """Find swing highs and lows"""
        highs = np.zeros_like(arr, dtype=bool)
        lows = np.zeros_like(arr, dtype=bool)
        
        for i in range(window, len(arr) - window):
            # Swing high: higher than window bars on both sides
            if arr[i] == np.max(arr[i-window:i+window+1]):
                highs[i] = True
            # Swing low: lower than window bars on both sides
            if arr[i] == np.min(arr[i-window:i+window+1]):
                lows[i] = True
        return highs, lows
    
    # Find swing points in price and RSI
    price_highs, price_lows = find_swing_points(close, window=3)
    rsi_highs, rsi_lows = find_swing_points(rsi_values, window=3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # Check for bullish divergence: price lower low, RSI higher low
            bullish_div = False
            if price_lows[i] and rsi_lows[i]:
                # Look back for previous swing low
                for j in range(i-20, i):
                    if price_lows[j] and rsi_lows[j]:
                        if close[i] < close[j] and rsi_values[i] > rsi_values[j]:
                            bullish_div = True
                        break
            
            # Check for bearish divergence: price higher high, RSI lower high
            bearish_div = False
            if price_highs[i] and rsi_highs[i]:
                # Look back for previous swing high
                for j in range(i-20, i):
                    if price_highs[j] and rsi_highs[j]:
                        if close[i] > close[j] and rsi_values[i] < rsi_values[j]:
                            bearish_div = True
                        break
            
            # LONG: bullish divergence + volume exhaustion + above EMA50
            if bullish_div and volume_exhaustion[i] and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish divergence + volume exhaustion + below EMA50
            elif bearish_div and volume_exhaustion[i] and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # EXIT LONG: RSI overbought or trend reversal
            if rsi_values[i] > 70 or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or trend reversal
            if rsi_values[i] < 30 or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals