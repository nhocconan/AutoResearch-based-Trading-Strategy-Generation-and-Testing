#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_1dTrend_Filter
Hypothesis: 6h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar avg). Uses ATR-based stoploss. Designed for low trade frequency (<30/year) to minimize fee drag. Works in bull/bear by following 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's Camarilla levels (using 1d OHLC)
    def camarilla_levels(high_arr, low_arr, close_arr):
        # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
        # We only need R1, S1, R3, S3 for breakout/fade logic
        R1 = close_arr + 1.1 * (high_arr - low_arr) / 12
        S1 = close_arr - 1.1 * (high_arr - low_arr) / 12
        R3 = close_arr + 1.1 * (high_arr - low_arr) / 6
        S3 = close_arr - 1.1 * (high_arr - low_arr) / 6
        return R1, S1, R3, S3
    
    R1_1d, S1_1d, R3_1d, S3_1d = camarilla_levels(high_1d, low_1d, close_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # ATR for stoploss (20-period)
    def atr(high_arr, low_arr, close_arr, period=20):
        tr = np.zeros_like(close_arr)
        atr_vals = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        tr[0] = high_arr[0] - low_arr[0]
        atr_vals[period-1] = np.mean(tr[1:period]) if period > 1 else tr[0]
        for i in range(period, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr_vals = atr(high, low, close, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need EMA50 (50), volume MA (20), ATR (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_vals[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above R1 in 1d uptrend with volume confirmation
            long_signal = (curr_close > R1_1d_aligned[i]) and \
                         (ema_50_1d_aligned[i] > close_1d[-1] if len(close_1d) > 0 else False) and \
                         volume_spike[i]
            # Short: price breaks below S1 in 1d downtrend with volume confirmation
            short_signal = (curr_close < S1_1d_aligned[i]) and \
                          (ema_50_1d_aligned[i] < close_1d[-1] if len(close_1d) > 0 else False) and \
                          volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend turns down OR stoploss hit
            if (curr_close < S1_1d_aligned[i]) or \
               (ema_50_1d_aligned[i] < close_1d[-1] if len(close_1d) > 0 else False) or \
               (curr_close < entry_price - 2.0 * atr_vals[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend turns up OR stoploss hit
            if (curr_close > R1_1d_aligned[i]) or \
               (ema_50_1d_aligned[i] > close_1d[-1] if len(close_1d) > 0 else False) or \
               (curr_close > entry_price + 2.0 * atr_vals[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0