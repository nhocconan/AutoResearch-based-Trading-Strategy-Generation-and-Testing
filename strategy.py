#!/usr/bin/env python3
"""
1d_williams_alligator_v1
Hypothesis: Williams Alligator on 1-day chart with 1-week trend filter and volume confirmation.
- Long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA) with price above both, 
  weekly uptrend (price > weekly SMA50), and volume expansion
- Short when jaws crosses below teeth with price below both, weekly downtrend, and volume expansion
- Uses Williams Alligator's smoothed moving averages to reduce whipsaw in ranging markets
- Weekly trend filter ensures alignment with higher timeframe momentum
- Designed for low trade frequency (10-30/year) to minimize fee drift
- Works in bull/bear via weekly trend filter and volume confirmation requirement
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_williams_alligator_v1"
timeframe = "1d"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - Williams Alligator uses SMMA"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    
    smma = np.full_like(data, np.nan, dtype=float)
    smma[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1-day
    # Jaws: 13-period SMMA of median price
    median_price = (high + low) / 2.0
    median_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        median_1d[i] = (df_1d['high'].iloc[i] + df_1d['low'].iloc[i]) / 2.0
    
    jaws = smma(median_1d, 13)  # Blue line
    teeth = smma(median_1d, 8)   # Red line
    lips = smma(median_1d, 5)    # Green line (not used in signals)
    
    # Weekly trend filter: price > weekly SMA50 for uptrend
    close_1w = df_1w['close'].values
    sma_50_1w = np.full_like(close_1w, np.nan, dtype=float)
    for i in range(50, len(close_1w)):
        sma_50_1w[i] = np.mean(close_1w[i-50:i])
    
    # Align indicators to daily timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume confirmation: 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        jaw = jaws_aligned[i]
        tooth = teeth_aligned[i]
        weekly_uptrend = price > sma_50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: jaws crosses below teeth OR volume drops significantly
            if jaw < tooth or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: jaws crosses above teeth OR volume drops significantly
            if jaw > tooth or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: jaws crosses above teeth with price above both, weekly uptrend, volume expansion
            if jaw > tooth and price > jaw and price > tooth and weekly_uptrend and vol_ratio > 1.8:
                position = 1
                signals[i] = 0.25
            # Enter short: jaws crosses below teeth with price below both, weekly downtrend, volume expansion
            elif jaw < tooth and price < jaw and price < tooth and not weekly_uptrend and vol_ratio > 1.8:
                position = -1
                signals[i] = -0.25
    
    return signals