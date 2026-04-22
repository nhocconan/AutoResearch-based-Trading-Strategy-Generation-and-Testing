#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot bias and volume confirmation.
# Uses weekly pivot levels for directional bias, Donchian breakouts for entry timing, and volume spike for confirmation.
# Long when price breaks above Donchian high in bullish bias (weekly pivot > previous weekly pivot) with volume spike.
# Short when price breaks below Donchian low in bearish bias (weekly pivot < previous weekly pivot) with volume spike.
# Exit on opposite Donchian touch or bias reversal.
# Designed for 6h timeframe to target 12-37 trades/year per symbol.
# Works in bull/bear via weekly pivot bias filter + volatility-based entry levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for weekly pivot calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot point (using Sunday's daily data as weekly proxy)
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    # We approximate weekly high/low/close using rolling window of 5 days (trading week)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Weekly bias: current weekly pivot vs previous weekly pivot
    weekly_bias = weekly_pivot - np.roll(weekly_pivot, 1)
    weekly_bias[0] = 0  # first value has no previous
    
    # Donchian channels (20-period) on 6f data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    # Align weekly bias to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + bullish bias + volume spike
            if (close[i] > donchian_high[i] and 
                weekly_bias_aligned[i] > 0 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + bearish bias + volume spike
            elif (close[i] < donchian_low[i] and 
                  weekly_bias_aligned[i] < 0 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on Donchian low touch or bias turns bearish
                if (close[i] < donchian_low[i] or weekly_bias_aligned[i] < 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Donchian high touch or bias turns bullish
                if (close[i] > donchian_high[i] or weekly_bias_aligned[i] > 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Bias_VolumeSpike"
timeframe = "6h"
leverage = 1.0