#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
    # Weekly pivot provides institutional bias for the week (from Monday open)
    # Donchian breakout captures momentum in direction of weekly bias
    # Volume spike (2x 20-period) confirms institutional participation
    # Works in bull/bear: breaks through key levels with weekly bias and volume
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot from previous week (weekly OHLC)
    # Use Monday open as week start approximation
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot point (standard calculation)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly bias: above/below pivot
    weekly_bias = np.where(weekly_close > weekly_pivot, 1, -1)  # 1=bullish, -1=bearish
    
    # Align weekly bias to 6h timeframe (using previous week's bias)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Load daily data for Donchian channels (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) channels on daily data
    # Upper channel: 20-day high
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-day low
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe (using previous day's channels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_bias_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with weekly bullish bias and volume spike
            if close[i] > donchian_high_aligned[i] and weekly_bias_aligned[i] == 1 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with weekly bearish bias and volume spike
            elif close[i] < donchian_low_aligned[i] and weekly_bias_aligned[i] == -1 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level
            if position == 1:
                if close[i] < donchian_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_20_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0