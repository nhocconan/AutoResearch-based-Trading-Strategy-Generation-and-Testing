#!/usr/bin/env python3
# [24967] 6h_1d_cci_trend_reversal_v1
# Hypothesis: 6-hour CCI(20) reversal with 1-day trend filter and volume confirmation.
# Long when CCI crosses above -100 from below with price > 1-day EMA200 and volume > 1.5x average.
# Short when CCI crosses below +100 from above with price < 1-day EMA200 and volume > 1.5x average.
# Exit when CCI crosses zero or volume drops below 1.2x average.
# Uses CCI mean reversion in trends, effective in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_trend_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        alpha = 2.0 / (200 + 1)
        ema_200_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_200_1d[i-1]
    
    # Calculate CCI(20)
    typical_price = (high + low + close) / 3.0
    tp_ma = np.full(n, np.nan)
    tp_mad = np.full(n, np.nan)
    cci = np.full(n, np.nan)
    
    for i in range(20, n):
        tp_ma[i] = np.mean(typical_price[i-20:i])
        deviation = np.abs(typical_price[i-20:i] - tp_ma[i])
        tp_mad[i] = np.mean(deviation)
        if tp_mad[i] > 0:
            cci[i] = (typical_price[i] - tp_ma[i]) / (0.015 * tp_mad[i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1-day EMA200 to 6-hour timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(cci[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: CCI crosses zero or volume drops below 1.2x average
            if cci[i] <= 0 or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: CCI crosses zero or volume drops below 1.2x average
            if cci[i] >= 0 or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: CCI crosses above -100 from below with uptrend and volume
            if i > 20 and cci[i-1] <= -100 and cci[i] > -100 and price > ema_200_1d_aligned[i] and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: CCI crosses below +100 from above with downtrend and volume
            elif i > 20 and cci[i-1] >= 100 and cci[i] < 100 and price < ema_200_1d_aligned[i] and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals