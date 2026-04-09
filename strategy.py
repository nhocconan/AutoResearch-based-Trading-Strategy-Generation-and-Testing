#!/usr/bin/env python3
# 12h_pullback_to_vwap_with_trend_filter_v1
# Hypothesis: In strong trends (identified by 1d EMA50 direction), price pulls back to VWAP on 12h timeframe before continuing the trend.
# Uses 1d EMA50 for trend filter, 12h VWAP for mean reversion entries, and volume confirmation.
# Works in both bull and bear markets by following the higher timeframe trend.
# Target: 15-30 trades per year (60-120 over 4 years) with low frequency to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_pullback_to_vwap_with_trend_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate VWAP on 12h (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    
    # Cumulative sums for VWAP
    pv_cumsum = np.zeros(n)
    vol_cumsum = np.zeros(n)
    
    pv_sum = 0.0
    vol_sum = 0.0
    for i in range(n):
        pv_sum += pv[i]
        vol_sum += volume[i]
        pv_cumsum[i] = pv_sum
        vol_cumsum[i] = vol_sum
    
    # VWAP = cumulative PV / cumulative volume
    vwap = np.full(n, np.nan)
    for i in range(n):
        if vol_cumsum[i] > 0:
            vwap[i] = pv_cumsum[i] / vol_cumsum[i]
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily for trend filter
    alpha = 2 / (50 + 1)
    ema50_1d = np.zeros(len(df_1d))
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(df_1d)):
        ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Trend: 1 if close > EMA50, -1 if close < EMA50
    trend_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if close_1d[i] > ema50_1d[i]:
            trend_1d[i] = 1
        elif close_1d[i] < ema50_1d[i]:
            trend_1d[i] = -1
        else:
            trend_1d[i] = 0  # rare case of exact equality
    
    # Align trend to 12h timeframe
    trend_12h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume confirmation: 20-period average
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
        if np.isnan(vwap[i]) or np.isnan(trend_12h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: trend changes to downtrend or price moves above VWAP (take profit)
            if trend_12h[i] == -1 or close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend changes to uptrend or price moves below VWAP (take profit)
            if trend_12h[i] == 1 or close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: uptrend + pullback to VWAP + volume
            if (trend_12h[i] == 1 and 
                high[i] >= vwap[i] * 0.995 and  # Allow small tolerance for touching VWAP
                low[i] <= vwap[i] * 1.005 and
                close[i] < vwap[i] and  # Close below VWAP for pullback entry
                vol_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: downtrend + pullback to VWAP + volume
            elif (trend_12h[i] == -1 and 
                  low[i] <= vwap[i] * 1.005 and  # Allow small tolerance for touching VWAP
                  high[i] >= vwap[i] * 0.995 and
                  close[i] > vwap[i] and  # Close above VWAP for pullback entry
                  vol_ok):
                position = -1
                signals[i] = -0.25
    
    return signals