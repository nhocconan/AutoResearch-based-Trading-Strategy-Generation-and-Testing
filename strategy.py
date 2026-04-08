#!/usr/bin/env python3
# 12h_donchian10_volume_trend_v1
# Hypothesis: 12h timeframe reduces trade frequency to avoid fee drag. Uses Donchian(10) breakout with volume confirmation and 1d EMA trend filter.
# Long when price breaks above Donchian(10) high, volume > 1.5x average, and price > 1d EMA50.
# Short when price breaks below Donchian(10) low, volume > 1.5x average, and price < 1d EMA50.
# Exit when price crosses the opposite Donchian band.
# Designed for 10-30 trades/year on 12h timeframe to minimize fee impact while capturing trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian10_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (10-period)
    lookback = 10
    dc_high = np.full(n, np.nan)
    dc_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        dc_high[i] = np.max(high[i-lookback+1:i+1])
        dc_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(lookback, vol_ma_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
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
            # Long entry: Price above Donchian high, volume surge, and price above 1d EMA50
            if (close[i] > dc_high[i] and 
                vol_surge[i] and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below Donchian low, volume surge, and price below 1d EMA50
            elif (close[i] < dc_low[i] and 
                  vol_surge[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals