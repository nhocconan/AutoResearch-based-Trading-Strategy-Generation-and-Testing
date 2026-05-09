#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Works in bull (breakouts continue) and bear (fades false breaks via EMA filter).
# Target: 20-40 trades/year, low turnover, avoids whipsaw.

name = "4h_Donchian20_1dEMA50_Volume"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    dc_high = high_series.rolling(window=20, min_periods=20).max().values
    dc_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian window
    
    for i in range(start_idx, n):
        # Skip if any data unavailable
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_1d_aligned[i]
        dc_high_val = dc_high[i]
        dc_low_val = dc_low[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price breaks above Donchian high + above 1d EMA50 + volume
            if close[i] > dc_high_val and close[i] > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian low + below 1d EMA50 + volume
            elif close[i] < dc_low_val and close[i] < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below Donchian low
            if close[i] < dc_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Donchian high
            if close[i] > dc_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals