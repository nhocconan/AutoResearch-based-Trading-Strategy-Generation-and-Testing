#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter_v1
Concept: Daily KAMA trend with RSI momentum and Choppiness regime filter.
- Long: KAMA trending up AND RSI > 50 AND Choppiness < 61.8 (trending regime)
- Short: KAMA trending down AND RSI < 50 AND Choppiness < 61.8 (trending regime)
- Exit: Opposite signal or Choppiness > 61.8 (range regime)
- Position sizing: 0.25
- Target: 20-50 trades/year (80-200 total over 4 years)
- Works in bull/bear: KAMA adapts to volatility, RSI filters momentum, Choppiness avoids range whipsaw
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Daily: KAMA trend indicator ===
    close = prices['close'].values
    # Efficiency Ratio: |price change - n periods ago| / sum of absolute changes
    change = np.abs(close - np.roll(close, 10))  # 10-period ER
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # placeholder for rolling sum
    
    # Proper ER calculation using pandas for rolling sum
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = close_series.diff().abs().rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === Daily: RSI(14) momentum ===
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # === Weekly: Choppiness Index regime filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14)
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[0:14])  # simple average for first 14
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of True Range over 14 periods
    tr_sum = np.zeros_like(tr)
    for i in range(13, len(tr)):
        if i == 13:
            tr_sum[i] = np.sum(tr[0:14])
        else:
            tr_sum[i] = tr_sum[i-1] - tr[i-14] + tr[i]
    
    # Choppiness Index
    chop = np.zeros_like(tr)
    for i in range(13, len(tr)):
        if tr_sum[i] > 0 and atr[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (atr[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Handle beginning values
    chop[:13] = 50
    
    # Align weekly Choppiness to daily
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop_aligned[i]
        
        # Skip if any value is invalid
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA trending up, RSI bullish, trending regime
            if close[i] > kama_val and rsi_val > 50 and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down, RSI bearish, trending regime
            elif close[i] < kama_val and rsi_val < 50 and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA cross down OR range regime
            if close[i] < kama_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA cross up OR range regime
            if close[i] > kama_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals