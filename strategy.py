#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_1dVolume_Spike_And_1wTrend_Filter
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 12h to capture trend direction.
Enters long when price crosses above KAMA with 1d volume spike confirmation and 1w uptrend.
Enters short when price crosses below KAMA with 1d volume spike confirmation and 1w downtrend.
Uses 1d volume spike (>2.0x 20-period MA) to filter false signals.
Designed to work in both bull and bear markets by following the adaptive trend.
Targets 12-37 trades per year to minimize fee drag while capturing meaningful moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume spike (>2.0x 20-period MA)
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_20)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA on 12h close
    # KAMA parameters: ER = 10, Fast = 2, Slow = 30
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # 1w trend filter
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike_1d_aligned[i]
        
        # Entry logic:
        # Long: Price crosses above KAMA with volume spike and 1w uptrend
        long_entry = vol_confirm and price_above_kama and not (close[i-1] > kama[i-1]) and trend_up
        # Short: Price crosses below KAMA with volume spike and 1w downtrend
        short_entry = vol_confirm and price_below_kama and not (close[i-1] < kama[i-1]) and trend_down
        
        # Exit logic: Opposite cross of KAMA
        long_exit = price_below_kama and (close[i-1] > kama[i-1])
        short_exit = price_above_kama and (close[i-1] < kama[i-1])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Trend_With_1dVolume_Spike_And_1wTrend_Filter"
timeframe = "12h"
leverage = 1.0