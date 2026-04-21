#!/usr/bin/env python3
"""
12h_HTF_1d_KAMA_RSI_ChopFilter_V1
Hypothesis: Use 1d KAMA direction (trend filter) + 12h RSI extremes for mean reversion entries, 
with 12h Choppiness Index regime filter (CHOP > 61.8 = range) to avoid trending markets. 
Exit on RSI mean reversion (RSI 40-60) or opposite extreme. 
Works in bull/bear: KAMA filters trend direction, RSI captures reversals in range markets (chop filter ensures ranging conditions). 
Discrete sizing 0.25 to minimize fee churn. Target 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for daily KAMA and Chop filter
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d KAMA (trend filter) ===
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d))
    change = np.insert(change, 0, np.nan)
    volatility = np.abs(np.diff(close_1d, 1))
    volatility = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    volatility = np.insert(volatility, 0, np.nan)
    er = change / volatility
    er = np.nan_to_num(er, nan=0.0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_dir = kama > np.roll(kama, 1)  # True if rising
    kama_dir[0] = False
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir.astype(float))
    
    # === 1d Choppiness Index (regime filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_1d = np.maximum(high_1d - low_1d, 
                        np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                   np.abs(low_1d - np.roll(close_1d, 1))))
    atr_1d[0] = np.nan
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum()
    high_low = pd.Series(high_1d).rolling(window=14, min_periods=14).max() - \
               pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / high_low) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # === 12h Indicators ===
    close = prices['close'].values
    
    # RSI (14-period)
    delta = np.diff(close)
    delta = np.insert(delta, 0, np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(chop_aligned[i]) 
            or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in ranging markets (Chop > 61.8)
        in_range = chop_aligned[i] > 61.8
        
        if not in_range:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        kama_up = kama_dir_aligned[i] > 0.5  # KAMA trending up
        
        if position == 0:
            # Long: RSI oversold (30) in ranging market, KAMA up (bullish bias)
            if rsi_val < 30 and kama_up:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (70) in ranging market, KAMA down (bearish bias)
            elif rsi_val > 70 and not kama_up:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI mean reversion (40-60) or opposite extreme
            if rsi_val > 40 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI mean reversion (40-60) or opposite extreme
            if rsi_val < 60 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_1d_KAMA_RSI_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0