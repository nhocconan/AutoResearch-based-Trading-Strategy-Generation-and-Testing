#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_With_1W_Volume_Filter
Hypothesis: Use KAMA on 1w trend for direction and KAMA on 1d for entry timing. Add weekly volume confirmation to reduce false signals. Works in both bull and bear markets by following strong weekly trends. Targets 10-20 trades/year with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and volume
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate KAMA for weekly trend
    def kama(close, length=10, fast=2, slow=30):
        dir = np.abs(np.diff(close, n=length))
        vol = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(vol != 0, dir / vol, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.full_like(close, np.nan, dtype=float)
        kama_out[length] = close[length]
        for i in range(length+1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    # Calculate KAMA for daily entry
    def kama_daily(close, length=10, fast=2, slow=30):
        dir = np.abs(np.diff(close, n=length))
        vol = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(vol != 0, dir / vol, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.full_like(close, np.nan, dtype=float)
        kama_out[length] = close[length]
        for i in range(length+1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    # Calculate weekly KAMA and volume average
    kama_1w = kama(close_1w, 10, 2, 30)
    vol_ma_1w = np.full(len(volume_1w), np.nan)
    for i in range(20, len(volume_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-20:i])
    
    # Calculate daily KAMA for entry
    kama_1d = kama_daily(close, 10, 2, 30)
    
    # Align weekly data to daily
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need weekly and daily data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(kama_1d[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/below KAMA
        weekly_uptrend = close[i] > kama_1w_aligned[i]
        weekly_downtrend = close[i] < kama_1w_aligned[i]
        
        # Weekly volume confirmation: current weekly volume > 1.5 * 20-week average
        # Need to get the weekly volume for the current week
        week_idx = i // (7*24*4)  # approximate weeks in 1d data
        if week_idx < len(volume_1w) and week_idx >= 20:
            vol_confirmed = volume_1w[week_idx] > 1.5 * vol_ma_1w[week_idx]
        else:
            vol_confirmed = False
        
        if position == 0:
            # Long entry: weekly uptrend, price above daily KAMA, volume confirmation
            if weekly_uptrend and close[i] > kama_1d[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend, price below daily KAMA, volume confirmation
            elif weekly_downtrend and close[i] < kama_1d[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: weekly trend changes or price crosses below daily KAMA
            if not weekly_uptrend or close[i] < kama_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend changes or price crosses above daily KAMA
            if not weekly_downtrend or close[i] > kama_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_KAMA_Trend_With_1W_Volume_Filter"
timeframe = "1d"
leverage = 1.0