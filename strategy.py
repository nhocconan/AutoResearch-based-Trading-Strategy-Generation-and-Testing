#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_v2
# Hypothesis: Daily Donchian breakouts with weekly trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, weekly price > weekly SMA(50), and volume > 1.5x average.
# Short when price breaks below Donchian(20) low, weekly price < weekly SMA(50), and volume > 1.5x average.
# Exit when price crosses opposite Donchian band.
# Target: 10-25 trades/year with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channel (20-period)
    lookback = 20
    dc_high = np.full(n, np.nan)
    dc_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        dc_high[i] = np.max(high[i-lookback+1:i+1])
        dc_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Weekly trend filter: SMA(50) on weekly closes
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_sma = np.full(len(weekly_close), np.nan)
    for i in range(49, len(weekly_close)):
        weekly_sma[i] = np.mean(weekly_close[i-49:i+1])
    weekly_sma_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma)
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(lookback, vol_ma_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(weekly_sma_aligned[i]) or np.isnan(vol_ma[i])):
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
            # Long entry: Price above Donchian high, weekly uptrend, volume surge
            if (close[i] > dc_high[i] and 
                close[i] > weekly_sma_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below Donchian low, weekly downtrend, volume surge
            elif (close[i] < dc_low[i] and 
                  close[i] < weekly_sma_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals