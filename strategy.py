#!/usr/bin/env python3
# 4h_price_channel_breakout_v1
# Hypothesis: Breakout of 4-hour Donchian channel (20-period) with volume confirmation and 1-day trend filter.
# Enters long when price breaks above upper band with volume > 1.5x average, and 1-day EMA50 rising.
# Enters short when price breaks below lower band with volume > 1.5x average, and 1-day EMA50 falling.
# Uses weekly trend filter to avoid counter-trend trades in strong weekly trends.
# Designed for 20-40 trades/year on 4h to avoid fee drag. Works in bull/bear via multi-timeframe alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_channel_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(period, n):
        highest_high[i] = np.max(high[i-period:i+1])
        lowest_low[i] = np.min(low[i-period:i+1])
    
    upper_band = highest_high
    lower_band = lowest_low
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= period:
            vol_sum -= volume[i - period]
            vol_count -= 1
        if vol_count == period:
            vol_avg[i] = vol_sum / period
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    ema50_1d[49] = np.mean(close_1d[:50])
    for i in range(50, len(close_1d)):
        ema50_1d[i] = (close_1d[i] * 2/51) + (ema50_1d[i-1] * 49/51)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1-week EMA50 for trend filter (avoid counter-trend)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema50_1w[i] = (close_1w[i] * 2/51) + (ema50_1w[i-1] * 49/51)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(period, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filters
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below lower band or trend turns down
            if close[i] < lower_band[i] or not uptrend_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper band or trend turns up
            if close[i] > upper_band[i] or not downtrend_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with volume and trend alignment
            if (close[i] > upper_band[i] and 
                vol_confirm and 
                uptrend_1d and 
                uptrend_1w):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume and trend alignment
            elif (close[i] < lower_band[i] and 
                  vol_confirm and 
                  downtrend_1d and 
                  downtrend_1w):
                position = -1
                signals[i] = -0.25
    
    return signals