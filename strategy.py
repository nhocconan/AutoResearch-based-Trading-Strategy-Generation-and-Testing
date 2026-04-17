#!/usr/bin/env python3
"""
4h_KAMA_Trend_Filter_V1
Strategy: 4h Kaufman Adaptive Moving Average (KAMA) with trend filter and volume confirmation.
Long: Price > KAMA(10,2,30) + volume > 1.3x 20-period avg
Short: Price < KAMA(10,2,30) + volume > 1.3x 20-period avg
Exit: Opposite condition
Position size: 0.25
Uses daily EMA34 as trend filter to avoid counter-trend trades.
Designed to work in both bull and bear markets by adapting to market noise.
Timeframe: 4h
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
    
    # Calculate KAMA on close
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        n = len(close)
        kama_out = np.full(n, np.nan)
        if n < er_length:
            return kama_out
        
        # Efficiency Ratio
        change = np.abs(close[er_length:] - close[:-er_length])
        volatility = np.sum(np.abs(np.diff(close[:er_length+1])) if len(close) >= er_length+1 else 0)
        er = np.zeros(n)
        for i in range(er_length, n):
            if volatility > 0:
                er[i] = change[i-er_length] / volatility
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # Initialize KAMA
        kama_out[er_length] = close[er_length]
        
        # Calculate KAMA
        for i in range(er_length + 1, n):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        
        return kama_out
    
    # Calculate KAMA
    kama_val = kama(close, 10, 2, 30)
    
    # Calculate daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume average (20-period)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(kama_val[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.3 * volume_ma20_4h_aligned[i])
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Entry signals
        if position == 0:
            # Long: Price > KAMA + volume filter + trend up
            if close[i] > kama_val[i] and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA + volume filter + trend down
            elif close[i] < kama_val[i] and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price < KAMA or trend down
            if close[i] < kama_val[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price > KAMA or trend up
            if close[i] > kama_val[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_Filter_V1"
timeframe = "4h"
leverage = 1.0