#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day KAMA trend filter and 12-hour RSI mean reversion.
# In trending markets (price > KAMA), use RSI extremes for mean-reversion entries.
# In ranging markets (price < KAMA), avoid entries to reduce whipsaw.
# Uses 1-day KAMA for trend detection and 12-hour RSI(14) for entry timing.
# Exits when RSI returns to neutral (50) or trend reverses.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_KAMA_Trend_RSI_MeanReversion"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # 1-day KAMA (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    # Efficiency Ratio
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = [np.nan] * len(close_1d)
    if len(close_1d) > 0:
        kama[0] = close_1d.iloc[0]
        for i in range(1, len(close_1d)):
            if not np.isnan(sc.iloc[i]):
                kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    kama = np.array(kama)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 12-hour RSI(14) for mean reversion
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Trend condition: price > KAMA (bullish trend)
    price_above_kama = close > kama_aligned
    # RSI conditions for mean reversion
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    rsi_neutral = (rsi >= 45) & (rsi <= 55)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or
            np.isnan(price_above_kama[i]) or
            np.isnan(rsi_oversold[i]) or np.isnan(rsi_overbought[i]) or np.isnan(rsi_neutral[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish trend + RSI oversold
            if price_above_kama[i] and rsi_oversold[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + RSI overbought
            elif (not price_above_kama[i]) and rsi_overbought[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral OR trend reverses
            if rsi_neutral[i] or (not price_above_kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral OR trend reverses
            if rsi_neutral[i] or price_above_kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals