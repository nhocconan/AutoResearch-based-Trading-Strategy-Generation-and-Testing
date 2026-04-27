#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop
Hypothesis: Use daily KAMA to establish trend, RSI(2) for mean-reversion entries, and Choppiness Index to filter ranging vs trending markets. Enter long in uptrend when RSI(2) < 10, short in downtrend when RSI(2) > 90. Exit when RSI(2) crosses 50. Designed to work in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly KAMA for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    # Efficiency Ratio
    change = abs(close_1w.diff(10))
    volatility = close_1w.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama = [close_1w.iloc[0]]
    for i in range(1, len(close_1w)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1w.iloc[i] - kama[-1]))
    kama_1w = np.array(kama)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI(2) for entry signals
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=2, min_periods=2).mean()
    avg_loss = loss.rolling(window=2, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14) for regime filter
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1))))
    atr_sum = atr.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 30  # need 30 for weekly KAMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: uptrend (price > weekly KAMA) and oversold RSI(2) in trending market
            if (close[i] > kama_1w_aligned[i] and 
                rsi[i] < 10 and 
                chop[i] < 61.8):  # trending market
                signals[i] = 0.25
                position = 1
            # Short: downtrend (price < weekly KAMA) and overbought RSI(2) in trending market
            elif (close[i] < kama_1w_aligned[i] and 
                  rsi[i] > 90 and 
                  chop[i] < 61.8):  # trending market
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI(2) crosses above 50 or trend changes
            if rsi[i] > 50 or close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI(2) crosses below 50 or trend changes
            if rsi[i] < 50 or close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Chop"
timeframe = "1d"
leverage = 1.0