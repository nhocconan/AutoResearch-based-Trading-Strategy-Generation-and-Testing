#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_v2
Hypothesis: Camarilla pivot levels from 1d chart act as strong support/resistance on 4h timeframe.
Price tends to reverse or bounce from these levels with confirmation from volume spike and RSI filter.
Uses 1d for structure (reduced noise) and 4h for timely entries. Works in both bull and bear markets.
Target: 20-30 trades per year (80-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_v2"
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
    
    # Get 1D data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # === CAMARILLA PIVOT LEVELS (based on previous 1d bar) ===
    # Calculate from previous 1d bar's OHLC
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    
    # First bar will have invalid data, but we'll handle with valid check
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    l3 = pivot + (range_val * 1.1 / 4)
    l4 = pivot + (range_val * 1.1 / 2)
    h3 = pivot - (range_val * 1.1 / 4)
    h4 = pivot - (range_val * 1.1 / 2)
    
    # Align to 4h timeframe (these levels are valid for the entire 1d bar)
    l3_4h = align_htf_to_ltf(prices, df_1d, l3)
    l4_4h = align_htf_to_ltf(prices, df_1d, l4)
    h3_4h = align_htf_to_ltf(prices, df_1d, h3)
    h4_4h = align_htf_to_ltf(prices, df_1d, h4)
    
    # === RSI FILTER (14-period on 1d) ===
    delta = pd.Series(daily_close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === VOLUME SPIKE (2x 20-period average on 4h) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid (first bar roll will have NaN)
        if (np.isnan(l3_4h[i]) or np.isnan(l4_4h[i]) or 
            np.isnan(h3_4h[i]) or np.isnan(h4_4h[i]) or
            np.isnan(rsi_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price near Camarilla levels (within 0.15% tolerance)
        near_l3 = abs(low[i] - l3_4h[i]) / l3_4h[i] < 0.0015
        near_l4 = abs(low[i] - l4_4h[i]) / l4_4h[i] < 0.0015
        near_h3 = abs(high[i] - h3_4h[i]) / h3_4h[i] < 0.0015
        near_h4 = abs(high[i] - h4_4h[i]) / h4_4h[i] < 0.0015
        
        # RSI conditions for reversal
        rsi_oversold = rsi_4h[i] < 30
        rsi_overbought = rsi_4h[i] > 70
        
        # Entry conditions with volume confirmation
        long_entry = (near_l3 or near_l4) and rsi_oversold and vol_spike[i]
        short_entry = (near_h3 or near_h4) and rsi_overbought and vol_spike[i]
        
        # Exit conditions: price moves back toward pivot or opposite signal
        pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
        long_exit = close[i] >= pivot_4h[i]  # Exit long when price reaches pivot
        short_exit = close[i] <= pivot_4h[i]  # Exit short when price reaches pivot
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals