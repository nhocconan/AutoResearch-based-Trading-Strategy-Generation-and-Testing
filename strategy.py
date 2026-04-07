#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Volume and ADX Filter
# Hypothesis: Donchian(20) breakouts capture institutional momentum. 
# Volume confirms participation, ADX>25 filters for trending markets.
# Works in bull (breakouts continue) and bear (breakdowns continue).
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_donchian20_volume_adx_v1"
timeframe = "4h"
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
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    dc_high = high_series.rolling(window=20, min_periods=20).max().values
    dc_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to Donchian low or ADX weakens or volume drops
            if (close[i] <= dc_low[i] or adx[i] < 20 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to Donchian high or ADX weakens or volume drops
            if (close[i] >= dc_high[i] or adx[i] < 20 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and ADX>25
            if (high[i] > dc_high[i] and close[i] > dc_high[i] and 
                vol_filter[i] and adx[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and ADX>25
            elif (low[i] < dc_low[i] and close[i] < dc_low[i] and 
                  vol_filter[i] and adx[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals