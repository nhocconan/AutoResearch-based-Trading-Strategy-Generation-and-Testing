#!/usr/bin/env python3
# 4h_donchian_volume_breakout_v1
# Hypothesis: Donchian(20) breakout with volume confirmation and ATR-based position sizing.
# Long when price breaks above 20-period high with volume > 1.5x average.
# Short when price breaks below 20-period low with volume > 1.5x average.
# Exit when price crosses the 20-period midpoint.
# Uses tight entry conditions to limit trades and reduce fee drag. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_breakout_v1"
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
    
    # Donchian channel (20-period high/low)
    donch_period = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=donch_period, min_periods=donch_period).max().values
    donch_low = low_series.rolling(window=donch_period, min_periods=donch_period).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=vol_ma_period, min_periods=vol_ma_period).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = donch_period  # Wait for Donchian to be fully formed
    
    for i in range(start_idx, n):
        # Skip if volume data not available
        if np.isnan(vol_ma[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below midpoint
            if close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above midpoint
            if close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high with volume surge
            if close[i] > donch_high[i] and vol_surge[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low with volume surge
            elif close[i] < donch_low[i] and vol_surge[i]:
                position = -1
                signals[i] = -0.25
    
    return signals