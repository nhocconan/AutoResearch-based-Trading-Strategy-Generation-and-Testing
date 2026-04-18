#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrendVolume
Hypothesis: Trade Donchian(20) breakouts on 4h with 1d trend filter and volume confirmation.
Enter long when 4h close breaks above 20-period high + 1d close > 1d SMA50 + volume > 1.5x 24-period average.
Enter short when 4h close breaks below 20-period low + 1d close < 1d SMA50 + volume > 1.5x 24-period average.
Exit when price crosses back through Donchian midpoint or trend reverses.
Designed to capture trends in bull markets and avoid whipsaw in sideways markets via trend filter.
Target: 20-40 trades/year via strict breakout conditions + trend filter.
Works in bull by following breakouts, in bear by avoiding false breaks via 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend filter
    sma_period = 50
    close_1d = df_1d['close'].values
    sma_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= sma_period:
        # Use pandas for efficient calculation
        sma_series = pd.Series(close_1d).rolling(window=sma_period, min_periods=sma_period).mean()
        sma_1d = sma_series.values
    
    # Align 1d SMA to 4h timeframe
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Donchian channels on 4h (20-period)
    donchian_period = 20
    high_4h = high
    low_4h = low
    
    # Calculate rolling max/min using pandas for efficiency
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_high = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Donchian midpoint for exit
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_period = 24
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=vol_period, min_periods=vol_period).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, sma_period, vol_period)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(sma_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + uptrend (price > SMA50) + volume
            if close[i] > donchian_high[i] and close[i] > sma_1d_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + downtrend (price < SMA50) + volume
            elif close[i] < donchian_low[i] and close[i] < sma_1d_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint or trend turns down
            if close[i] < donchian_mid[i] or close[i] < sma_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint or trend turns up
            if close[i] > donchian_mid[i] or close[i] > sma_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrendVolume"
timeframe = "4h"
leverage = 1.0