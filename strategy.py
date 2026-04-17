#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Uses Donchian breakouts for momentum, weekly pivot (PP) for trend filter, volume spike for confirmation.
# Designed to work in bull (breakouts above weekly PP) and bear (breakouts below weekly PP).
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot point (PP)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: PP = (high + low + close) / 3
    weekly_pp = (high_1w + low_1w + close_1w) / 3.0
    
    # Calculate Donchian(20) channels on 6h data
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly PP to 6h
    weekly_pp_6h = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    # Volume filter: current volume > 1.5 * 50-period average (reduces trades)
    volume_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need volume MA50 and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pp_6h[i]) or 
            np.isnan(high_max20[i]) or 
            np.isnan(low_min20[i]) or 
            np.isnan(volume_ma50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to balance trades)
        volume_filter = volume[i] > (1.5 * volume_ma50[i])
        
        # Donchian breakout conditions
        breakout_up = high[i] > high_max20[i-1]  # Use previous bar's high_max20
        breakout_down = low[i] < low_min20[i-1]  # Use previous bar's low_min20
        
        # Price relative to weekly pivot point
        price_above_pp = close[i] > weekly_pp_6h[i]
        price_below_pp = close[i] < weekly_pp_6h[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and above weekly PP
            if (breakout_up and price_above_pp and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume and below weekly PP
            elif (breakout_down and price_below_pp and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Donchian low OR below weekly PP
            if (low[i] < low_min20[i]) or (close[i] < weekly_pp_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Donchian high OR above weekly PP
            if (high[i] > high_max20[i]) or (close[i] > weekly_pp_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPP_Volume"
timeframe = "6h"
leverage = 1.0