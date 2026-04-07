#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend Filter
Long when price breaks above 20-day Donchian high with weekly uptrend
Short when price breaks below 20-day Donchian low with weekly downtrend
Exit when price crosses 10-day SMA or weekly trend reverses
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 10-day SMA for exit ===
    close_series = pd.Series(close)
    sma_10 = close_series.rolling(window=10, min_periods=10).mean().values
    
    # === Weekly trend filter (HMA) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        hma_1w = np.full(n, np.nan)
    else:
        # Calculate HMA on weekly close
        weekly_close = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(arr, n):
            if len(arr) < n:
                return np.full(len(arr), np.nan)
            weights = np.arange(1, n + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        n_hma = 21
        half_n = n_hma // 2
        sqrt_n = int(np.sqrt(n_hma))
        
        wma_full = wma(weekly_close, n_hma)
        wma_half = wma(weekly_close, half_n)
        wma_diff = 2 * wma_half - wma_full
        hma_raw = wma(wma_diff, sqrt_n)
        
        # Pad to match weekly_close length
        hma_1w_raw = np.full(len(weekly_close), np.nan)
        if len(hma_raw) > 0:
            start_idx = len(weekly_close) - len(hma_raw)
            hma_1w_raw[start_idx:] = hma_raw
        
        # Align to daily and shift by 1 week for completed bar only
        hma_1w = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # === Donchian channels (20-day) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(sma_10[i]) or np.isnan(hma_1w[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 10-day SMA or weekly trend turns down
            if close[i] < sma_10[i] or hma_1w[i] < hma_1w[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 10-day SMA or weekly trend turns up
            if close[i] > sma_10[i] or hma_1w[i] > hma_1w[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Weekly trend filter: only trade in direction of weekly trend
            weekly_uptrend = hma_1w[i] > hma_1w[i-1]
            weekly_downtrend = hma_1w[i] < hma_1w[i-1]
            
            # Entry: Donchian breakout with weekly trend alignment
            if weekly_uptrend and close[i] > donch_high[i]:
                position = 1
                signals[i] = 0.25
            elif weekly_downtrend and close[i] < donch_low[i]:
                position = -1
                signals[i] = -0.25
    
    return signals