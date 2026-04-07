#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h KAMA with RSI and Chop Filter
# Hypothesis: KAMA adapts to market noise, reducing whipsaws in choppy markets.
# RSI filters for momentum extremes, while Choppiness Index identifies ranging vs trending regimes.
# In trending markets (CHOP < 38.2), follow KAMA direction. In ranging markets (CHOP > 61.8), fade RSI extremes.
# Works in both bull and bear markets by adapting to regime.
# Targets 12-37 trades/year with disciplined entries to avoid overtrading.

name = "12h_kama_rsi_chop_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)  # 2-period EMA
    slow_sc = 2 / (30 + 1) # 30-period EMA
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC) for KAMA
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Fix: volatility needs to be rolling sum of absolute changes
    volatility = pd.Series(np.abs(np.diff(close))).rolling(window=10, min_periods=1).sum().values
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align with change
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(13, np.nan), rsi[13:]])  # align with original close
    
    # Choppiness Index (14-period)
    atr = np.abs(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))))
    atr[0] = high[0] - low[0]  # first ATR
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) != 0, 
                    100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14), 
                    50)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # align with original close
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1w trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h_w = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup for indicators
        # Skip if required data not available
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema50_12h[i]) or 
            np.isnan(ema50_12h_w[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: trend change or opposite signal
            if is_trending and close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            elif is_ranging and rsi[i] > 70:  # overbought in range
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: trend change or opposite signal
            if is_trending and close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            elif is_ranging and rsi[i] < 30:  # oversold in range
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine trend direction from higher timeframes
            trend_up = close[i] > ema50_12h[i] and close[i] > ema50_12h_w[i]
            trend_down = close[i] < ema50_12h[i] and close[i] < ema50_12h_w[i]
            
            if is_trending:
                # Trending market: follow KAMA direction
                if trend_up and close[i] > kama[i]:
                    position = 1
                    signals[i] = 0.25
                elif trend_down and close[i] < kama[i]:
                    position = -1
                    signals[i] = -0.25
            elif is_ranging:
                # Ranging market: fade RSI extremes
                if rsi[i] < 30:  # oversold
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70:  # overbought
                    position = -1
                    signals[i] = -0.25
    
    return signals