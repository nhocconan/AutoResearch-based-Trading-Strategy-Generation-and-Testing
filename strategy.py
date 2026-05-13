#!/usr/bin/env python3
"""
4h_RSI_Divergence_Bullish_Bearish
Hypothesis: RSI divergences on 4H timeframe combined with 1D trend filter and volume confirmation provide high-probability entries in both bull and bear markets.
Bullish divergence: price makes lower low, RSI makes higher low. Bearish divergence: price makes higher high, RSI makes lower high.
Enter on divergence confirmation with price action and volume spike. Exit on opposite divergence or trend change.
Target: 20-40 trades/year per symbol.
"""

name = "4h_RSI_Divergence_Bullish_Bearish"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1D trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    # Detect bullish and bearish divergences
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    # Look for price lows and RSI lows for bullish divergence
    for i in range(20, n-10):
        # Find local price low in window [i-10, i+10]
        if low[i] == np.min(low[i-10:i+11]):
            # Look back for previous low
            for j in range(i-20, i-10):
                if low[j] == np.min(low[j-10:j+11]) and j >= 10:
                    if low[i] < low[j] and rsi[i] > rsi[j]:  # Lower low, higher RSI low
                        bullish_div[i] = True
                    break
    
    # Look for price highs and RSI highs for bearish divergence
    for i in range(20, n-10):
        # Find local price high in window [i-10, i+10]
        if high[i] == np.max(high[i-10:i+11]):
            # Look back for previous high
            for j in range(i-20, i-10):
                if high[j] == np.max(high[j-10:j+11]) and j >= 10:
                    if high[i] > high[j] and rsi[i] < rsi[j]:  # Higher high, lower RSI high
                        bearish_div[i] = True
                    break
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Get values
        bull_div = bullish_div[i]
        bear_div = bearish_div[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: bullish divergence + 1D uptrend + volume confirmation
            if bull_div and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish divergence + 1D downtrend + volume confirmation
            elif bear_div and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish divergence or 1D trend turns down
            if bear_div or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish divergence or 1D trend turns up
            if bull_div or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals