#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation.
# Uses Donchian channels for breakout signals, weekly pivot for trend filter, volume for confirmation.
# Designed to work in bull (breakouts with trend) and bear (mean reversion via pivot rejects).
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from daily data
    # Weekly high/low/close from last 5 trading days
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly Pivot Point (PP) = (H + L + C) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1 = 2*PP - L, S1 = 2*PP - H
    weekly_r1 = 2 * weekly_pp - weekly_low
    weekly_s1 = 2 * weekly_pp - weekly_high
    
    # Align weekly pivot to 6h
    weekly_pp_6h = align_htf_to_ltf(prices, df_1d, weekly_pp)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Calculate Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need Donchian and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pp_6h[i]) or 
            np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to balance frequency)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Price relative to Donchian channels
        price_above_donchian_high = close[i] > donchian_high[i]
        price_below_donchian_low = close[i] < donchian_low[i]
        
        # Price relative to weekly pivot levels
        price_above_pp = close[i] > weekly_pp_6h[i]
        price_below_pp = close[i] < weekly_pp_6h[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and above weekly PP
            if (price_above_donchian_high and price_above_pp and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume and below weekly PP
            elif (price_below_donchian_low and price_below_pp and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian low OR below weekly PP
            if (close[i] < donchian_low[i]) or (close[i] < weekly_pp_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian high OR above weekly PP
            if (close[i] > donchian_high[i]) or (close[i] > weekly_pp_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0