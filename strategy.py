#!/usr/bin/env python3
# 4h_camarilla_pivot_volume_v2
# Hypothesis: Uses 1d Camarilla pivot levels with volume spike confirmation and 1d EMA trend filter.
# Trades only when price approaches key pivot levels (L3, L3, H3, H4) with volume confirmation.
# Designed for low-frequency, high-conviction trades in both bull and bear markets.
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1. 1d Camarilla pivot levels (from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    L3 = close_1d - (high_1d - low_1d) * 1.1 / 8
    L4 = close_1d - (high_1d - low_1d) * 1.1 / 6
    H3 = close_1d + (high_1d - low_1d) * 1.1 / 6
    H4 = close_1d + (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    
    # 2. 1d EMA trend filter (21-period)
    ema21_1d = np.zeros(len(close_1d))
    if len(close_1d) >= 21:
        ema21_1d[0] = close_1d[0]
        alpha = 2 / (21 + 1)
        for i in range(1, len(close_1d)):
            ema21_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema21_1d[i-1]
    
    # Trend: 1 if close > EMA21, -1 if close < EMA21
    trend_1d = np.where(close_1d > ema21_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 3. Volume confirmation (24-period average = 24*4h = 96h ~ 4 days)
    vol_ma_24 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(H4_aligned[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 24-period average
        vol_ok = volume[i] > vol_ma_24[i] * 1.8
        
        if position == 1:  # Long position
            # Exit: price moves below L3 OR trend turns bearish
            if close[i] < L3_aligned[i] or trend_1d_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above H3 OR trend turns bullish
            if close[i] > H3_aligned[i] or trend_1d_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price approaches L4 from above with volume and bullish trend
            if (close[i] <= L4_aligned[i] * 1.002 and  # Within 0.2% of L4
                close[i] >= L3_aligned[i] * 0.998 and  # Above L3
                vol_ok and 
                trend_1d_aligned[i] == 1):
                position = 1
                signals[i] = 0.25
            # Enter short: price approaches H4 from below with volume and bearish trend
            elif (close[i] >= H4_aligned[i] * 0.998 and  # Within 0.2% of H4
                  close[i] <= H3_aligned[i] * 1.002 and  # Below H3
                  vol_ok and 
                  trend_1d_aligned[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals