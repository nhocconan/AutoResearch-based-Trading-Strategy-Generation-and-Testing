#!/usr/bin/env python3
"""
6h Daily Close Reversion with 1-day Trend Filter and Volume Confirmation
Mean reversion strategy: enter long when price closes below previous day's low with volume confirmation and above daily EMA20 (uptrend), short when price closes above previous day's high with volume confirmation and below daily EMA20 (downtrend)
Designed to work in both bull and bear markets by capturing short-term reversals within the prevailing daily trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily levels and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily data to 6h
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume spike detection (1.5x 24-period average - 4 days worth)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        prev_high = high_1d_aligned[i]
        prev_low = low_1d_aligned[i]
        ema_trend = ema_20_1d_aligned[i]
        
        if position == 0:
            # Long: close below previous day's low + volume spike + above daily EMA20 (uptrend)
            if (price < prev_low and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: close above previous day's high + volume spike + below daily EMA20 (downtrend)
            elif (price > prev_high and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes back above previous day's low or trend reversal
            if price > prev_low or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes back below previous day's high or trend reversal
            if price < prev_high or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyCloseReversion_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0