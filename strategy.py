#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Regime
Hypothesis: Uses weekly trend direction (KAMA) with daily RSI mean-reversion in non-trending regimes.
In bull markets: weekly KAMA up + daily RSI < 30 → long.
In bear markets: weekly KAMA down + daily RSI > 70 → short.
In ranging markets: RSI extremes with mean reversion.
Uses Chop index to filter trending vs ranging regimes.
Target: 10-25 trades/year on 1d (40-100 total over 4 years).
"""

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
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on weekly
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = abs(pd.Series(close_1w).diff(10))
    volatility = pd.Series(close_1w).diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2))**2  # fast=2, slow=30
    sc = sc.fillna(0.01)**2
    kama = np.zeros(len(close_1w))
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Get daily data for RSI and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate RSI(14) on daily
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # Calculate Chop Index(14) on daily
    true_range = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    true_range[0] = high_1d[0] - low_1d[0]
    atr14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean()
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14 * 14 / (max_high - min_low)) / np.log10(14)
    chop = chop.fillna(50)
    
    # Align all signals to 1d timeframe (prices index)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        price = close[i]
        
        # Determine regime based on Chop
        # Chop > 61.8 = ranging, Chop < 38.2 = trending
        if chop_val > 61.8:  # Ranging market
            # Mean reversion at RSI extremes
            if rsi_val < 30 and position != 1:
                position = 1
                signals[i] = position_size
            elif rsi_val > 70 and position != -1:
                position = -1
                signals[i] = -position_size
            elif position == 1 and rsi_val > 50:  # Exit long on RSI recovery
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi_val < 50:  # Exit short on RSI decline
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
        else:  # Trending market
            # Follow weekly KAMA trend with RSI pullback entries
            if kama_val > price and rsi_val < 40 and position != 1:  # Pullback in uptrend
                position = 1
                signals[i] = position_size
            elif kama_val < price and rsi_val > 60 and position != -1:  # Pullback in downtrend
                position = -1
                signals[i] = -position_size
            elif position == 1 and (kama_val <= price or rsi_val > 70):  # Exit long
                position = 0
                signals[i] = 0.0
            elif position == -1 and (kama_val >= price or rsi_val < 30):  # Exit short
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_KAMA_RSI_Regime"
timeframe = "1d"
leverage = 1.0