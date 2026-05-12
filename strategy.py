#!/usr/bin/env python3
"""
1d_RSI_Divergence_VolumeTrend
Hypothesis: On daily timeframe, bullish/bearish RSI divergence (price making higher lows/lower highs while RSI makes opposite) combined with volume trend confirmation (increasing volume on up moves/decreasing on down moves) captures trend reversals and continuations. Works in bull markets via bullish divergences leading uptrends, and in bear markets via bearish divergences leading downtrends. Uses 14-day RSI and 20-day volume trend filter to avoid false signals.
"""

name = "1d_RSI_Divergence_VolumeTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume trend: 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_trend = volume > vol_ema  # volume above its EMA = increasing trend
    
    # Detect swing points for divergence
    def find_swing_points(arr, window=5):
        """Find local highs and lows using rolling window"""
        highs = np.zeros_like(arr, dtype=bool)
        lows = np.zeros_like(arr, dtype=bool)
        
        for i in range(window, len(arr) - window):
            if arr[i] == np.max(arr[i-window:i+window+1]):
                highs[i] = True
            if arr[i] == np.min(arr[i-window:i+window+1]):
                lows[i] = True
        return highs, lows
    
    # Find price and RSI swing points
    price_highs, price_lows = find_swing_points(close, window=3)
    rsi_highs, rsi_lows = find_swing_points(rsi, window=3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if np.isnan(rsi[i]) or np.isnan(volume_trend[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Check for bullish divergence: price makes higher low, RSI makes lower low
            bullish_div = False
            if price_lows[i] and rsi_lows[i]:
                # Look back for previous lows
                for j in range(max(0, i-20), i):
                    if price_lows[j] and rsi_lows[j]:
                        if close[i] > close[j] and rsi[i] < rsi[j]:
                            bullish_div = True
                        break
            
            # Check for bearish divergence: price makes lower high, RSI makes higher high
            bearish_div = False
            if price_highs[i] and rsi_highs[i]:
                # Look back for previous highs
                for j in range(max(0, i-20), i):
                    if price_highs[j] and rsi_highs[j]:
                        if close[i] < close[j] and rsi[i] > rsi[j]:
                            bearish_div = True
                        break
            
            # Enter long on bullish divergence with volume confirmation
            if bullish_div and volume_trend[i]:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish divergence with volume confirmation
            elif bearish_div and volume_trend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Exit long on bearish divergence or volume trend reversal
            if (price_highs[i] and rsi_highs[i] and 
                any(close[i] < close[j] and rsi[i] > rsi[j] 
                    for j in range(max(0, i-20), i) if price_highs[j] and rsi_highs[j])) or \
               not volume_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on bullish divergence or volume trend reversal
            if (price_lows[i] and rsi_lows[i] and 
                any(close[i] > close[j] and rsi[i] < rsi[j] 
                    for j in range(max(0, i-20), i) if price_lows[j] and rsi_lows[j])) or \
               not volume_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals