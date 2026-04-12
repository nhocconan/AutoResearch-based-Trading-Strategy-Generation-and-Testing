#!/usr/bin/env python3
"""
4h_1D_Camarilla_Breakout_V3
Hypothesis: Breakout of daily Camarilla H3/L3 levels with volume confirmation and EMA trend filter.
Goes long when price breaks above H3 with volume > 1.5x average and EMA(50) > EMA(200).
Goes short when price breaks below L3 with volume > 1.5x average and EMA(50) < EMA(200).
Exits when price returns to the daily pivot level. Uses discrete position sizing (0.30) to limit drawdown.
Designed to work in both bull and bear markets by following institutional levels with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1D_Camarilla_Breakout_V3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivots and EMAs
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # === CAMARILLA PIVOT LEVELS (based on previous daily bar) ===
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    H3 = pivot + (range_val * 1.1 / 4)
    L3 = pivot - (range_val * 1.1 / 4)
    
    # Align to 4h timeframe
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === EMA TREND FILTER (50 and 200 on 1d) ===
    close_1d = pd.Series(daily_close)
    ema_50 = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_4h = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === VOLUME FILTER (1.5x 20-period average) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or 
            np.isnan(pivot_4h[i]) or np.isnan(ema_50_4h[i]) or
            np.isnan(ema_200_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Breakout conditions
        long_breakout = high[i] > H3_4h[i]
        short_breakout = low[i] < L3_4h[i]
        
        # Trend alignment
        uptrend = ema_50_4h[i] > ema_200_4h[i]
        downtrend = ema_50_4h[i] < ema_200_4h[i]
        
        # Entry conditions
        long_entry = long_breakout and uptrend and vol_filter[i]
        short_entry = short_breakout and downtrend and vol_filter[i]
        
        # Exit when price returns to pivot
        long_exit = position == 1 and close[i] <= pivot_4h[i]
        short_exit = position == -1 and close[i] >= pivot_4h[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.30
        elif long_exit or short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals