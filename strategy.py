#!/usr/bin/env python3
"""
Hypothesis: 4-hour MACD histogram with daily volume confirmation and weekly trend filter.
Enters long when MACD histogram turns positive with above-average volume and weekly uptrend.
Enters short when MACD histogram turns negative with above-average volume and weekly downtrend.
Uses weekly timeframe for trend structure to reduce noise and avoid false signals.
Designed to work in both bull and bear markets by following the weekly trend while using
MACD for momentum confirmation and volume for conviction. Target: 20-40 trades/year per
symbol to minimize fee drag and avoid overtrading.
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
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 4h MACD
    # MACD Line: 12-period EMA - 26-period EMA
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_12 - ema_26
    
    # Signal Line: 9-period EMA of MACD Line
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # MACD Histogram: MACD Line - Signal Line
    macd_hist = macd_line - signal_line
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need MACD components, volume MA, and weekly EMA
    start_idx = max(26, 9, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(macd_hist[i]) or np.isnan(macd_hist[i-1]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_20_1w_aligned[i]
        
        # Current MACD Histogram
        macd_hist_now = macd_hist[i]
        macd_hist_prev = macd_hist[i-1]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # MACD Histogram signals: crossing zero
        macd_hist_cross_up = macd_hist_prev <= 0 and macd_hist_now > 0
        macd_hist_cross_down = macd_hist_prev >= 0 and macd_hist_now < 0
        
        # Entry conditions
        if position == 0:
            # Long: MACD histogram crosses above zero with volume + weekly uptrend
            if macd_hist_cross_up and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: MACD histogram crosses below zero with volume + weekly downtrend
            elif macd_hist_cross_down and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: MACD histogram crosses below zero or weekly trend turns down
            if macd_hist_cross_down or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: MACD histogram crosses above zero or weekly trend turns up
            if macd_hist_cross_up or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_MACD_Histogram_1dVolume_1wTrend"
timeframe = "4h"
leverage = 1.0