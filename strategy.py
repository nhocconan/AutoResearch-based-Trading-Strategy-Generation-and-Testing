#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d VWAP trend filter + volume confirmation (2x average volume).
# Donchian channels identify breakout points, 1d VWAP confirms intraday trend alignment,
# and volume spikes validate momentum. Designed to capture strong trends in both bull and bear markets
# while filtering false breakouts in low-volume or ranging conditions. Target: 50-150 total trades over 4 years.
name = "4h_Donchian20_1dVWAP_Trend_Volume"
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
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1-day VWAP: sum(price * volume) / sum(volume) for the day
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_array = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_array)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    # For true rolling window, we need to reset every 20 periods
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_high_20[i] = np.max(high[i-20:i])
        lowest_low_20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or np.isnan(vwap_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        vwap = vwap_1d_aligned[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian AND price > 1d VWAP (uptrend) AND volume > 2x average
            if close[i] > upper_channel and close[i] > vwap and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian AND price < 1d VWAP (downtrend) AND volume > 2x average
            elif close[i] < lower_channel and close[i] < vwap and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR price < 1d VWAP (trend change)
            if close[i] < lower_channel or close[i] < vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR price > 1d VWAP (trend change)
            if close[i] > upper_channel or close[i] > vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals