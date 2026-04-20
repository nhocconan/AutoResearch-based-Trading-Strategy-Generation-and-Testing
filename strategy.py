#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_Filtered_Breakout
# Hypothesis: On 1d timeframe, trade breakouts from weekly KAMA-based channels with volume confirmation.
# Uses weekly KAMA to define trend direction and volatility bands. Targets 15-25 trades per year.
# Works in bull/bear via trend-aligned breakouts and volatility filtering.

name = "1d_1w_KAMA_Trend_Filtered_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA ( Kaufman Adaptive Moving Average )
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Efficiency ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w))
    er = np.zeros_like(close_1w)
    er[1:] = change[1:] / (np.abs(volatility).rolling(window=10, min_periods=1).sum() + 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Weekly ATR for volatility bands
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Upper and lower bands (KAMA ± 1.5 * ATR)
    upper_band = kama + 1.5 * atr_1w
    lower_band = kama - 1.5 * atr_1w
    
    # Align weekly levels to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(kama_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above upper band, volume surge, and price above KAMA (uptrend)
            if (close[i] > upper_aligned[i] and 
                volume[i] > 1.8 * volume_ma[i] and
                close[i] > kama_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below lower band, volume surge, and price below KAMA (downtrend)
            elif (close[i] < lower_aligned[i] and 
                  volume[i] > 1.8 * volume_ma[i] and
                  close[i] < kama_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA or touches lower band
            if close[i] < kama_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or touches upper band
            if close[i] > kama_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals