#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d VWAP trend filter and volume confirmation
# This strategy trades breakouts of the 12h Donchian channel with trend alignment from
# 1d VWAP and volume confirmation. It works in both bull and bear markets by following
# the trend direction on higher timeframe. Uses discrete position sizing (0.30) to
# balance return and minimize transaction costs.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for VWAP trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    
    # Align VWAP to 12h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 12h Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to allow for Donchian calculation
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above Donchian high + above 1d VWAP + volume spike
            if close[i] > donchian_high[i] and close[i] > vwap_1d_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below Donchian low + below 1d VWAP + volume spike
            elif close[i] < donchian_low[i] and close[i] < vwap_1d_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Price crosses Donchian midpoint in opposite direction
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if position == 1:
                # Exit long: Close below Donchian midpoint
                if close[i] < donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                # Exit short: Close above Donchian midpoint
                if close[i] > donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_Breakout_1dVWAP_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0