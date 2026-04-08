#!/usr/bin/env python3
# 4h_donchian_breakout_volume_v1
# Hypothesis: Breakouts from Donchian channel (20-period) with volume confirmation (>1.5x 20-period average volume).
# Long when price breaks above upper band with volume surge; short when breaks below lower band with volume surge.
# Exit when price crosses the middle line (mean of upper/lower) or volume drops below average.
# Uses 12h timeframe for trend filter: only take long if 12h SMA50 slope > 0, short if < 0.
# Designed to capture strong trending moves with low trade frequency to minimize fee drag.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    dc_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(dc_period-1, n):
        upper[i] = np.max(high[i-dc_period+1:i+1])
        lower[i] = np.min(low[i-dc_period+1:i+1])
    middle = (upper + lower) / 2.0
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # 12h SMA50 slope for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    sma_period = 50
    sma50_12h = pd.Series(close_12h).rolling(window=sma_period, min_periods=sma_period).mean().values
    sma50_slope_12h = np.full(len(close_12h), np.nan)
    for i in range(sma_period, len(close_12h)):
        if not np.isnan(sma50_12h[i]) and not np.isnan(sma50_12h[i-1]):
            sma50_slope_12h[i] = sma50_12h[i] - sma50_12h[i-1]
    sma50_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, sma50_slope_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(dc_period, vol_ma_period, sma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(sma50_slope_12h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below middle line or volume drops below average
            if close[i] < middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above middle line or volume drops below average
            if close[i] > middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above upper band, volume surge, 12h SMA50 slope positive
            if (close[i] > upper[i] and 
                vol_surge[i] and 
                sma50_slope_12h_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower band, volume surge, 12h SMA50 slope negative
            elif (close[i] < lower[i] and 
                  vol_surge[i] and 
                  sma50_slope_12h_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals