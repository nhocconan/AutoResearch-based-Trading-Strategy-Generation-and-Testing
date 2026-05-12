#!/usr/bin/env python3
# 6h_1d_1w_RSI_Trend_Divergence
# Hypothesis: Uses 1d RSI divergence with 1w trend filter on 6h timeframe to capture reversals in both bull and bear markets.
# Enters long when 6h price makes higher low while 1d RSI makes lower low (bullish divergence) and 1w trend is up.
# Enters short when 6h price makes lower high while 1d RSI makes higher high (bearish divergence) and 1w trend is down.
# Uses 6h volume confirmation to avoid false signals. Designed for low trade frequency (~50-150 total trades over 4 years).

name = "6h_1d_1w_RSI_Trend_Divergence"
timeframe = "6h"
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
    
    # 6h volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily RSI(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Weekly trend: EMA50 on 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate price swings for divergence detection
    # We'll look for swings over 3-period windows
    def find_swing_points(arr, window=3):
        """Find local minima and maxima"""
        n = len(arr)
        mins = np.zeros(n, dtype=bool)
        maxs = np.zeros(n, dtype=bool)
        
        for i in range(window, n - window):
            if arr[i] == np.min(arr[i-window:i+window+1]):
                mins[i] = True
            if arr[i] == np.max(arr[i-window:i+window+1]):
                maxs[i] = True
        return mins, maxs
    
    # Find swing points on 6h price and 1d RSI
    price_lows, price_highs = find_swing_points(close, window=3)
    rsi_lows, rsi_highs = find_swing_points(rsi_1d_aligned, window=3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(rsi_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: price makes higher low, RSI makes lower low
            bullish_div = False
            if price_lows[i] and rsi_lows[i]:
                # Look back for previous lows
                for j in range(max(10, i-50), i):
                    if price_lows[j] and rsi_lows[j]:
                        if close[i] > close[j] and rsi_1d_aligned[i] < rsi_1d_aligned[j]:
                            bullish_div = True
                            break
            
            # Bearish divergence: price makes lower high, RSI makes higher high
            bearish_div = False
            if price_highs[i] and rsi_highs[i]:
                # Look back for previous highs
                for j in range(max(10, i-50), i):
                    if price_highs[j] and rsi_highs[j]:
                        if close[i] < close[j] and rsi_1d_aligned[i] > rsi_1d_aligned[j]:
                            bearish_div = True
                            break
            
            # LONG: Bullish divergence + 1w uptrend + volume spike
            if bullish_div and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence + 1w downtrend + volume spike
            elif bearish_div and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence OR price closes below 1w EMA50
            bearish_div = False
            if price_highs[i] and rsi_highs[i]:
                for j in range(max(10, i-50), i):
                    if price_highs[j] and rsi_highs[j]:
                        if close[i] < close[j] and rsi_1d_aligned[i] > rsi_1d_aligned[j]:
                            bearish_div = True
                            break
            
            if bearish_div or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence OR price closes above 1w EMA50
            bullish_div = False
            if price_lows[i] and rsi_lows[i]:
                for j in range(max(10, i-50), i):
                    if price_lows[j] and rsi_lows[j]:
                        if close[i] > close[j] and rsi_1d_aligned[i] < rsi_1d_aligned[j]:
                            bullish_div = True
                            break
            
            if bullish_div or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals