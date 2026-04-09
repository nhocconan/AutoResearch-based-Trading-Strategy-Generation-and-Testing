#!/usr/bin/env python3
# 1d_kdj_reversal_v1
# Hypothesis: Uses 1-day KDJ (Stochastic) oscillator to catch reversals in BTC/ETH/SOL.
# In bear markets (2025+), reversals from oversold (K<20) and overbought (K>80) levels offer edge.
# Confirmed with weekly trend filter to avoid counter-trend trades and volume surge for confirmation.
# Target: 15-30 trades/year (60-120 total over 4 years) to stay within limits.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kdj_reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = np.zeros(len(df_1w))
    ema20_1w[0] = close_1w[0]
    for i in range(1, len(df_1w)):
        ema20_1w[i] = (2/21) * close_1w[i] + (19/21) * ema20_1w[i-1]
    trend_1w = np.where(close_1w > ema20_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_20_1w)
    
    # Calculate daily KDJ (9,3,3)
    if len(high) < 9:
        return np.zeros(n)
    
    lowest_low = np.zeros(n)
    highest_high = np.zeros(n)
    for i in range(n):
        if i < 8:
            lowest_low[i] = np.nan
            highest_high[i] = np.nan
        else:
            lowest_low[i] = np.min(low[i-8:i+1])
            highest_high[i] = np.max(high[i-8:i+1])
    
    rsv = np.zeros(n)
    for i in range(n):
        if highest_high[i] == lowest_low[i]:
            rsv[i] = 50
        else:
            rsv[i] = (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i]) * 100
    
    k = np.zeros(n)
    d = np.zeros(n)
    j = np.zeros(n)
    k[0] = 50
    d[0] = 50
    for i in range(1, n):
        if np.isnan(rsv[i]):
            k[i] = k[i-1]
            d[i] = d[i-1]
        else:
            k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
            d[i] = (2/3) * d[i-1] + (1/3) * k[i]
        j[i] = 3 * k[i] - 2 * d[i]
    
    # Volume confirmation: 20-day average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(j[i]) or 
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: KDJ overbought (K>80) or trend turns bearish
            if k[i] > 80 or trend_1w_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KDJ oversold (K<20) or trend turns bullish
            if k[i] < 20 or trend_1w_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: KDJ oversold (K<20) with volume and bullish weekly trend
            if (k[i] < 20 and 
                vol_ok and 
                trend_1w_aligned[i] == 1):
                position = 1
                signals[i] = 0.25
            # Enter short: KDJ overbought (K>80) with volume and bearish weekly trend
            elif (k[i] > 80 and 
                  vol_ok and 
                  trend_1w_aligned[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals