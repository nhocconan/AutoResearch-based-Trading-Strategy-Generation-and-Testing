#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price reversal at daily support/resistance zones with volume confirmation
# Uses daily pivot points (S1/R1) as key levels where price often reverses
# In trending markets, breaks through these levels signal continuation
# In ranging markets, bounces off these levels offer mean-reversion opportunities
# Works in both bull/bear by requiring volume confirmation and using volatility filter
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points and support/resistance levels
    # Pivot = (H + L + C) / 3
    # S1 = (2 * Pivot) - H
    # R1 = (2 * Pivot) - L
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    s1_1d = (2 * pivot_1d) - high_1d
    r1_1d = (2 * pivot_1d) - low_1d
    
    # Forward fill to get the most recent daily pivot levels
    pivot_1d = pd.Series(pivot_1d).ffill().values
    s1_1d = pd.Series(s1_1d).ffill().values
    r1_1d = pd.Series(r1_1d).ffill().values
    
    # Align daily pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # Volatility filter: avoid trading when volatility is too low
    # Use daily ATR(14) normalized by price
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_percent_1d = (atr_14_1d / close_1d) * 100
    atr_percent_aligned = align_htf_to_ltf(prices, df_1d, atr_percent_1d)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # for 20-period volume average and 14-period ATR
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(atr_percent_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        atr_percent = atr_percent_aligned[i]
        
        # Only trade when volatility is reasonable (avoid choppy low-volatility periods)
        if atr_percent < 0.5 or atr_percent > 5.0:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if (price > r1_aligned[i] and vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below S1 with volume confirmation
            elif (price < s1_aligned[i] and vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below pivot OR S1
            if price < pivot_aligned[i] or price < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above pivot OR R1
            if price > pivot_aligned[i] or price > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Daily_Pivot_Reversal_Volume_Filter"
timeframe = "12h"
leverage = 1.0