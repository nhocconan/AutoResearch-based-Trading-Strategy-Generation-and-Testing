#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian Breakout with 1d Volume Profile and Volume Confirmation
# Hypothesis: Donchian channel breakouts capture breakout momentum, while 1d volume profile
# identifies institutional interest areas. Volume confirmation ensures breakouts have
# participation. Works in bull (breakouts continue) and bear (breakdowns continue) markets.
# Target: 15-40 trades/year (60-160 over 4 years) with strict entry conditions.
name = "6h_donchian_breakout_1d_volume_profile_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for volume profile
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian Channel (20-period) on 6h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # 1-day Volume Profile: Calculate value area (VAH/VAL) from 20-day period
    # Simplified: use 20-day VWAP as proxy for institutional interest
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_array = vwap_1d.values
    vwap_1d_6h = align_htf_to_ltf(prices, df_1d, vwap_1d_array)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vwap_1d_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle or breaks below Donchian low with volume
            if close[i] < donchian_middle[i] or (close[i] < donchian_low[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle or breaks above Donchian high with volume
            if close[i] > donchian_middle[i] or (close[i] > donchian_high[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long breakout: price breaks above Donchian high with volume
                if close[i] > donchian_high[i] and close[i] > vwap_1d_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price breaks below Donchian low with volume
                elif close[i] < donchian_low[i] and close[i] < vwap_1d_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals