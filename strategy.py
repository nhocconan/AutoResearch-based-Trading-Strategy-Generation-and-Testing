#!/usr/bin/env python3
"""
12h_KAMA_Direction_1dRSI_ChopFilter
Hypothesis: Uses KAMA (adaptive trend) on 12h for trend direction, combined with 1d RSI extremes and 1d Choppiness Index regime filter.
- Long: KAMA upward (close > KAMA) AND 1d RSI < 30 (oversold) AND 1d Choppiness > 61.8 (ranging market -> mean reversion)
- Short: KAMA downward (close < KAMA) AND 1d RSI > 70 (overbought) AND 1d Choppiness > 61.8 (ranging market -> mean reversion)
- Exit: Opposite KAMA cross OR RSI returns to neutral (40-60) OR chop regime ends (Choppiness < 38.2 -> trending)
Works in both bull and bear markets by using adaptive trend (KAMA) and fading extremes in ranging regimes.
Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year).
Volume is not used as primary filter to avoid overtrading; instead relies on regime and extreme conditions.
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
    
    # Get 1d data for RSI and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h KAMA (adaptive trend)
    # Efficiency Ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series - close_series.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants: fastest EMA=2 (ER=1), slowest EMA=30 (ER=0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Handle NaN/inf
    sc = sc.fillna(0.001)  # default to slow EMA when ER is NaN
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50)  # neutral when undefined
    
    # Calculate 1d Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR(1), 14) / (log10(highest_high - lowest_low) * log10(14)))
    tr1 = np.maximum(df_1d['high'].values - df_1d['low'].values,
                     np.maximum(abs(df_1d['high'].values - close_1d.shift(1)),
                                abs(df_1d['low'].values - close_1d.shift(1))))
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum()  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    chop_denom = np.log10(highest_high - lowest_low) * np.log10(14)
    chop_denom = chop_denom.replace(0, np.nan)  # avoid division by zero
    chop_1d = 100 * np.log10(sum_atr1 / chop_denom)
    chop_1d = chop_1d.fillna(50)  # neutral when undefined
    
    # Align 1d indicators to 12h timeframe (completed bars only)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama.values)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need KAMA (10), RSI (14), Chop (14)
    start_idx = max(14, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        
        if position == 0:
            # Look for entry: KAMA direction + RSI extreme + Chop regime (ranging)
            # Long: KAMA up (close > KAMA) AND RSI < 30 (oversold) AND Chop > 61.8 (ranging -> mean reversion)
            long_condition = (close_val > kama_val) and (rsi_val < 30) and (chop_val > 61.8)
            # Short: KAMA down (close < KAMA) AND RSI > 70 (overbought) AND Chop > 61.8 (ranging -> mean reversion)
            short_condition = (close_val < kama_val) and (rsi_val > 70) and (chop_val > 61.8)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when KAMA turns down OR RSI returns to neutral OR chop regime ends (trending)
            exit_condition = (close_val < kama_val) or (rsi_val >= 40) or (chop_val < 38.2)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when KAMA turns up OR RSI returns to neutral OR chop regime ends (trending)
            exit_condition = (close_val > kama_val) or (rsi_val <= 60) or (chop_val < 38.2)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_Direction_1dRSI_ChopFilter"
timeframe = "12h"
leverage = 1.0