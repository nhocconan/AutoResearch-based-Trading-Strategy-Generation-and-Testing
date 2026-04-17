#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d VWAP filter and volume confirmation.
# Donchian(20) on 12h: upper/lower bands from highest high/lowest low of last 20 bars.
# VWAP on 1d: volume-weighted average price from 1d data, acts as dynamic support/resistance.
# Enter long when price breaks above Donchian upper with volume and above 1d VWAP.
# Enter short when price breaks below Donchian lower with volume and below 1d VWAP.
# Exit when price crosses back below/above 1d VWAP or Donchian middle.
# Designed for low turnover (target: 15-30 trades/year) with strong trend capture.
# Works in bull trends (breakouts) and bear trends (breakdowns via VWAP rejection).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d VWAP: typical price * volume, cumulative sum
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.where(vwap_den != 0, vwap_num / vwap_den, 0)
    
    # Align 1d VWAP to 12h timeframe
    vwap_12h = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average (strict to reduce trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vwap_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (very strict)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Price relative to Donchian bands
        price_above_upper = close[i] > donchian_high[i]
        price_below_lower = close[i] < donchian_low[i]
        
        # Price relative to 1d VWAP
        price_above_vwap = close[i] > vwap_12h[i]
        price_below_vwap = close[i] < vwap_12h[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume and above VWAP
            if (price_above_upper and price_above_vwap and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume and below VWAP
            elif (price_below_lower and price_below_vwap and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below VWAP OR below Donchian middle
            if (price_below_vwap) or (close[i] < donchian_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above VWAP OR above Donchian middle
            if (price_above_vwap) or (close[i] > donchian_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVWAP_Volume"
timeframe = "12h"
leverage = 1.0