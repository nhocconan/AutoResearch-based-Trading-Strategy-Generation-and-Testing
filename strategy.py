#!/usr/bin/env python3
# 1d_weekly_breakout_volume_v1
# Hypothesis: Uses weekly trend (price above/below weekly SMA50) as primary direction filter,
# daily Donchian(20) breakout as entry trigger, and volume confirmation (1.5x 20-day avg) to filter false breakouts.
# Weekly trend ensures alignment with higher timeframe momentum, reducing counter-trend trades.
# Donchian breakouts capture momentum bursts; volume confirmation adds validity.
# Designed for low-frequency, high-conviction trades to minimize fee drag.
# Target: 15-25 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian(20) for breakout signals
    donch_period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(donch_period-1, n):
        highest_high[i] = np.max(high[i-donch_period+1:i+1])
        lowest_low[i] = np.min(low[i-donch_period+1:i+1])
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get weekly data for trend filter (SMA50)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    sma50_weekly = pd.Series(close_weekly).rolling(window=50, min_periods=50).mean().values
    sma50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_period, vol_ma_period, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(sma50_weekly_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below weekly SMA50 (trend change) or Donchian low breaks
            if close[i] < sma50_weekly_aligned[i] or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above weekly SMA50 (trend change) or Donchian high breaks
            if close[i] > sma50_weekly_aligned[i] or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above weekly SMA50 (uptrend), breaks above Donchian high, volume surge
            if (close[i] > sma50_weekly_aligned[i] and 
                close[i] > highest_high[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below weekly SMA50 (downtrend), breaks below Donchian low, volume surge
            elif (close[i] < sma50_weekly_aligned[i] and 
                  close[i] < lowest_low[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals