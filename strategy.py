#!/usr/bin/env python3
# 4h_cci_adx_multiplier_v1
# Hypothesis: Uses CCI to identify overbought/oversold conditions and ADX for trend strength on 4h timeframe.
# Long when CCI crosses above -100 with ADX > 25 (trending up); short when CCI crosses below 100 with ADX > 25 (trending down).
# Includes 1-day trend filter using EMA50 to avoid counter-trend trades.
# Designed to work in both bull and bear markets by trading with the trend only when momentum is strong.
# Target: 20-30 trades/year (80-120 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_adx_multiplier_v1"
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
    
    # 1. CCI (20 period)
    typical_price = (high + low + close) / 3.0
    tp_ma = np.zeros(n)
    tp_mean_dev = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - 19)
        tp_slice = typical_price[start_idx:i+1]
        tp_ma[i] = np.mean(tp_slice)
        tp_mean_dev[i] = np.mean(np.abs(tp_slice - tp_ma[i]))
    
    cci = np.zeros(n)
    for i in range(n):
        if tp_mean_dev[i] > 0:
            cci[i] = (typical_price[i] - tp_ma[i]) / (0.015 * tp_mean_dev[i])
        else:
            cci[i] = 0.0
    
    # 2. ADX (14 period)
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Subsequent values using Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, plus_dm14 / tr14 * 100, 0)
    minus_di14 = np.where(tr14 != 0, minus_dm14 / tr14 * 100, 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # 3. 1-day EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[:] = np.nan
    
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(cci[i]) or np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 or ADX weakens
            if cci[i] < 100 and cci[i-1] >= 100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 or ADX weakens
            if cci[i] > -100 and cci[i-1] <= -100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: CCI crosses above -100 with strong ADX and price above 1-day EMA50
            if (cci[i] > -100 and cci[i-1] <= -100 and 
                adx[i] > 25 and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: CCI crosses below +100 with strong ADX and price below 1-day EMA50
            elif (cci[i] < 100 and cci[i-1] >= 100 and 
                  adx[i] > 25 and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals