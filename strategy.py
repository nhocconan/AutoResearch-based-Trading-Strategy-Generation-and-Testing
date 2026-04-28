#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
# Donchian breakouts capture trend continuation, especially effective in volatile crypto markets.
# ATR filter (current ATR > 1.5x 20-period average) ensures trades occur in high-volatility regimes,
# avoiding choppy sideways markets where breakouts fail. Volume confirmation (1.5x 20-period average)
# validates breakout strength. Works in bull markets (catching uptrends via upper band breaks) and
# bear markets (catching downtrends via lower band breaks). Targets 20-50 trades/year with discrete
# position sizing to minimize fee drag.

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
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d True Range and ATR(20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range: max(high-low, abs(high-previous_close), abs(low-previous_close))
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan  # First value has no previous close
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(20) - simple moving average of TR
    atr_20_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_1d = pd.Series(atr_20_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter: current ATR > 1.5x 20-period average ATR
    atr_filter = atr_20_1d > (atr_ma_20_1d * 1.5)
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian and volume calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = high[i] > high_max[i-1]   # Break above upper band
        breakdown_down = low[i] < low_min[i-1]  # Break below lower band
        
        # Entry conditions with filters
        long_entry = breakout_up and volume_filter[i] and atr_filter_aligned[i]
        short_entry = breakdown_down and volume_filter[i] and atr_filter_aligned[i]
        
        # Exit conditions: opposite breakout or volatility collapse
        long_exit = breakdown_down or (not atr_filter_aligned[i])
        short_exit = breakout_up or (not atr_filter_aligned[i])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_1dATRVolatilityFilter_Volume"
timeframe = "4h"
leverage = 1.0