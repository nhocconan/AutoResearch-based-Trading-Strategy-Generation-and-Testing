#!/usr/bin/env python3
"""
4h_1D_KAMA_RSI_CHOP_V1
Hypothesis: Uses Kaufman's Adaptive Moving Average (KAMA) for trend direction, RSI for momentum filtering, and Choppiness Index as a regime filter to avoid whipsaws. Designed for low-frequency, high-probability entries in both bull and bear markets by requiring alignment of trend, momentum, and market regime. Targets 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1D_KAMA_RSI_CHOP_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # KAMA (4h) - trend direction
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * 0.59 + 0.01) ** 2
    kama = [np.nan] * len(close)
    kama[9] = close[9]
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    
    # RSI (4h) - momentum filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (1d) - regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.maximum(high_1d - low_1d,
                        np.maximum(abs(high_1d - np.roll(close_1d, 1)),
                                   abs(low_1d - np.roll(close_1d, 1))))
    atr_1d[0] = high_1d[0] - low_1d[0]
    sum_atr14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum()
    highest_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(sum_atr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50).values
    
    # Align 1d Choppiness to 4h (wait for daily close)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend: price relative to KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # Momentum: RSI levels
        rsi_overbought = rsi[i] > 60
        rsi_oversold = rsi[i] < 40
        
        # Regime: Choppiness Index
        # Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (trend follow)
        chopping = chop_aligned[i] > 61.8
        trending = chop_aligned[i] < 38.2
        
        # Entry conditions
        long_entry = above_kama and rsi_oversold and chopping  # Buy dip in range
        short_entry = below_kama and rsi_overbought and chopping  # Sell rally in range
        
        # Exit conditions: reverse signal or regime change to trending
        long_exit = (below_kama or rsi_overbought or trending)
        short_exit = (above_kama or rsi_oversold or trending)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals