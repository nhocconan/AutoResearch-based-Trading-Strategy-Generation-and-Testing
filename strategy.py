#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: 1d strategy using KAMA trend direction, RSI extremes, and Choppiness Index regime filter.
# KAMA adapts to market noise, providing reliable trend direction. RSI < 30 or > 70 identifies overextended conditions for mean reversion.
# Choppiness Index > 61.8 confirms ranging market (mean reversion favorable). Works in both bull/bear by fading extremes in ranging regimes.
# Discrete position sizing (±0.25) to minimize fee churn. Target: 30-100 total trades over 4 years (7-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for indicators (primary timeframe)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # KAMA trend direction (ER=10, fast=2, slow=30)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    kama_dir = np.where(close > kama, 1, -1)  # 1=above KAMA (bullish bias), -1=below (bearish bias)
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index(14)
    atr = np.maximum(high_s - low_s, np.maximum(high_s - close_s.shift(), close_s.shift() - low_s))
    atr = atr.rolling(window=14, min_periods=14).sum()
    highest_high = high_s.rolling(window=14, min_periods=14).max()
    lowest_low = low_s.rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        if np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) OR chop < 38.2 (trending regime)
            if rsi[i] > 50 or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 OR chop < 38.2
            if rsi[i] < 50 or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only trade in ranging regime: chop > 61.8
            if chop[i] > 61.8:
                # Long: RSI < 30 (oversold) AND price above KAMA (bullish bias)
                if rsi[i] < 30 and kama_dir[i] == 1:
                    position = 1
                    signals[i] = 0.25
                # Short: RSI > 70 (overbought) AND price below KAMA (bearish bias)
                elif rsi[i] > 70 and kama_dir[i] == -1:
                    position = -1
                    signals[i] = -0.25
    
    return signals