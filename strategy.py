#!/usr/bin/env python3
"""
4h_1d_VWAP_Trend_Signal
Hypothesis: Trade VWAP deviation from 1-day VWAP with 4h VWAP trend filter. 
Long when price > 1d VWAP and 4h VWAP rising; short when price < 1d VWAP and 4h VWAP falling.
VWAP acts as a dynamic mean and trend filter, working in both bull (buy dips) and bear (sell rallies).
Target: 80-150 total trades over 4 years with position size 0.25.
Uses VWAP for institutional alignment and mean reversion with trend filter to avoid chop.
"""

name = "4h_1d_VWAP_Trend_Signal"
timeframe = "4h"
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
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate daily VWAP (Volume Weighted Average Price)
    typical_price_daily = (df_daily['high'].values + df_daily['low'].values + df_daily['close'].values) / 3.0
    vol_daily = df_daily['volume'].values
    
    # Cumulative VWAP for daily
    cum_vol_price_daily = np.cumsum(typical_price_daily * vol_daily)
    cum_vol_daily = np.cumsum(vol_daily)
    vwap_daily = np.divide(cum_vol_price_daily, cum_vol_daily, out=np.full_like(cum_vol_price_daily, np.nan), where=cum_vol_daily!=0)
    vwap_daily_aligned = align_htf_to_ltf(prices, df_daily, vwap_daily)
    
    # Calculate 4h VWAP for trend filter
    typical_price = (high + low + close) / 3.0
    vol_price = typical_price * volume
    
    # Rolling 20-period VWAP for 4h trend (enough lookback for trend)
    window = 20
    cum_vol_price = np.zeros(n)
    cum_vol = np.zeros(n)
    vwap_4h = np.full(n, np.nan)
    
    for i in range(n):
        if i == 0:
            cum_vol_price[i] = vol_price[i]
            cum_vol[i] = volume[i]
        else:
            cum_vol_price[i] = cum_vol_price[i-1] + vol_price[i]
            cum_vol[i] = cum_vol[i-1] + volume[i]
        
        # Maintain rolling window
        if i >= window:
            cum_vol_price[i] -= vol_price[i-window]
            cum_vol[i] -= volume[i-window]
        
        if cum_vol[i] > 0:
            vwap_4h[i] = cum_vol_price[i] / cum_vol[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure VWAP calculations are stable
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_4h[i]) or np.isnan(vwap_daily_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above daily VWAP AND 4h VWAP rising
            if close[i] > vwap_daily_aligned[i] and vwap_4h[i] > vwap_4h[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price below daily VWAP AND 4h VWAP falling
            elif close[i] < vwap_daily_aligned[i] and vwap_4h[i] < vwap_4h[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below daily VWAP OR 4h VWAP falls
            if close[i] < vwap_daily_aligned[i] or vwap_4h[i] < vwap_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above daily VWAP OR 4h VWAP rises
            if close[i] > vwap_daily_aligned[i] or vwap_4h[i] > vwap_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals