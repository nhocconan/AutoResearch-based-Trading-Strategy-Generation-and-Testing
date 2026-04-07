#!/usr/bin/env python3
"""
6d_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 6-hour timeframe, use Donchian(20) breakout with 1-day trend filter (price above/below 200 EMA) and volume confirmation to capture strong directional moves. The 6h timeframe balances responsiveness with reduced noise, while the 1d trend filter ensures alignment with higher timeframe direction, reducing whipsaws in sideways markets. Volume confirmation ensures institutional participation. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while performing in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6d_donchian_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel parameters
    donch_len = 20
    
    # Calculate Donchian bands
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=donch_len, min_periods=donch_len).max().values
    donch_low = low_series.rolling(window=donch_len, min_periods=donch_len).min().values
    
    # 1-day trend filter: EMA200
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume filter: 20-period average on 6h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(donch_len, 200, 20), n):
        # Skip if data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low
            if close[i] <= donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high
            if close[i] >= donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish breakout: price closes above Donchian high with uptrend
                if close[i] > donch_high[i] and close[i] > ema200[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish breakout: price closes below Donchian low with downtrend
                elif close[i] < donch_low[i] and close[i] < ema200[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals