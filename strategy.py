#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d close > weekly pivot point AND volume > 1.5x 20-period average
# - Short when price breaks below Donchian(20) low AND 1d close < weekly pivot point AND volume > 1.5x 20-period average
# - Exit when price crosses Donchian(10) midpoint (mean reversion)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakouts capture momentum; weekly pivot provides HTF bias; volume filter avoids false breakouts
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_weekly_pivot_donchian_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h Donchian channels (20-period for entry, 10-period for exit)
    def highest_high(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def lowest_low(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    # Donchian(20) for entry
    hh20 = highest_high(high, 20)
    ll20 = lowest_low(low, 20)
    # Donchian(10) for exit (midpoint)
    hh10 = highest_high(high, 10)
    ll10 = lowest_low(low, 10)
    donchian_mid10 = (hh10 + ll10) / 2.0
    
    # Pre-compute 6h volume average (20-period)
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma20 = rolling_mean(volume, 20)
    
    # Pre-compute 1d weekly pivot point (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We'll approximate using daily data: weekly pivot = (monthly high + monthly low + monthly close) / 3
    # But simpler: use prior 5-day high, low, close (approximates weekly)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close (5-day lookback)
    weekly_high = rolling_max(high_1d, 5)
    weekly_low = rolling_min(low_1d, 5)
    weekly_close = close_1d  # Use daily close as proxy for weekly close
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align HTF indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hh20[i]) or np.isnan(ll20[i]) or np.isnan(vol_ma20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_mid10[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_confirm = volume[i] > 1.5 * vol_ma20[i]
            
            # Long conditions: break above Donchian(20) high AND 1d close > weekly pivot AND volume confirmation
            if (close[i] > hh20[i] and 
                close_1d[min(i // 24, len(close_1d)-1)] > weekly_pivot_aligned[i] and 
                volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short conditions: break below Donchian(20) low AND 1d close < weekly pivot AND volume confirmation
            elif (close[i] < ll20[i] and 
                  close_1d[min(i // 24, len(close_1d)-1)] < weekly_pivot_aligned[i] and 
                  volume_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian(10) midpoint (mean reversion)
            exit_long = (position == 1 and close[i] < donchian_mid10[i])
            exit_short = (position == -1 and close[i] > donchian_mid10[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals