#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1w pivot direction filter and volume confirmation
    # Donchian breakouts capture momentum in trending markets
    # Weekly pivot direction (price vs weekly pivot) filters for institutional bias
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in both bull and bear: breakouts in trends, pivot acts as dynamic support/resistance
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for pivot calculation (weekly pivot = (H+L+C)/3)
    df_1w = get_htf_data(prices, '1w')
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    weekly_pivot_values = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_values)
    
    # Donchian Channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + price above weekly pivot + volume spike
            if close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + price below weekly pivot + volume spike
            elif close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian band (mean reversion within channel)
            if position == 1:
                if close[i] < donchian_low[i]:  # Reverse to opposite band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i]:  # Reverse to opposite band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0