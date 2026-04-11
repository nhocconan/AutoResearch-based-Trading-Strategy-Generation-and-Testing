#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Strategy: 1d KAMA (Kaufman Adaptive Moving Average) for trend direction with RSI mean-reversion and 1w chop regime filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets. RSI identifies overbought/oversold conditions for mean-reversion entries. 1w chop filter avoids trend-following in ranging markets. Works in both bull and bear by combining trend-following (KAMA direction) with mean-reversion (RSI extremes) under appropriate volatility regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w chop regime: Choppy Index (CHOP) - high values indicate ranging market
    # Chop = 100 * log10(sum(ATR1) / (max(high) - min(low))) / log10(n)
    # We'll use a simplified version: ATR(14) / (max(high,14) - min(low,14)) * 100
    # Chop > 61.8 = ranging (favor mean reversion), Chop < 38.2 = trending (favor trend following)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR for 1w
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.abs(high_1w[0] - low_1w[0])], tr2])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate max(high) and min(low) over 14 periods for chop denominator
    max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    # Chop value: higher = more choppy/ranging
    chop_1w = (atr_1w / range_14) * 100
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # KAMA (Kaufman Adaptive Moving Average) for 1d trend direction
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    change = np.concatenate([[0]*10, change])  # align with original length
    
    # Volatility = sum of absolute changes over 10 periods
    volatility = np.sum(np.abs(np.diff(close, k=1)), axis=0) if len(close) > 1 else 0
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    volatility = np.concatenate([[0]*9, volatility[9:]]) if len(volatility) > 9 else np.zeros_like(close)
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI for mean-reversion signals
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])  # align with original length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop regime filter: Chop > 50 indicates ranging market (favor mean reversion)
        # Chop < 50 indicates trending market (favor trend following)
        is_ranging = chop_1w_aligned[i] > 50
        is_trending = chop_1w_aligned[i] <= 50
        
        # KAMA trend direction
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI mean-reversion levels
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Entry logic: Combine trend direction with mean-reversion in appropriate regime
        if is_ranging:
            # In ranging markets, look for mean-reversion at RSI extremes
            if rsi_oversold and price_above_kama and position != 1:
                # Oversold but price above KAMA - potential bounce up
                position = 1
                signals[i] = 0.25
            elif rsi_overbought and price_below_kama and position != -1:
                # Overbought but price below KAMA - potential drop down
                position = -1
                signals[i] = -0.25
        else:
            # In trending markets, follow KAMA direction
            if price_above_kama and not rsi_overbought and position != 1:
                # Uptrend and not overbought - go long
                position = 1
                signals[i] = 0.25
            elif price_below_kama and not rsi_oversold and position != -1:
                # Downtrend and not oversold - go short
                position = -1
                signals[i] = -0.25
        
        # Exit conditions
        if position == 1 and (rsi[i] > 70 or close[i] < kama[i]):
            # Exit long when overbought or price crosses below KAMA
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 30 or close[i] > kama[i]):
            # Exit short when oversold or price crosses above KAMA
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals