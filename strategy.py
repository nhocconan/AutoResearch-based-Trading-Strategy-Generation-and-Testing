#!/usr/bin/env python3
# 4h_price_channel_breakout_v1
# Hypothesis: Price channel breakouts with volume confirmation work in both bull and bear markets.
# Long when price breaks above Donchian(20) high with volume > 1.3x 20-period average.
# Short when price breaks below Donchian(20) low with volume > 1.3x 20-period average.
# Exit when price crosses the opposite Donchian band.
# Uses volume filter to reduce false breakouts and maintain low trade frequency.
# Target: 25-40 trades/year with strict entry conditions.

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
    
    # Donchian channel (20-period)
    lookback = 20
    dc_high = np.full(n, np.nan)
    dc_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        dc_high[i] = np.max(high[i-lookback+1:i+1])
        dc_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume filter: 1.3x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.3 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = lookback + vol_ma_period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below Donchian low
            if close[i] < dc_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Donchian high
            if close[i] > dc_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above Donchian high with volume surge
            if close[i] > dc_high[i] and vol_surge[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below Donchian low with volume surge
            elif close[i] < dc_low[i] and vol_surge[i]:
                position = -1
                signals[i] = -0.25
    
    return signals