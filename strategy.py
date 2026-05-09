#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with volume and volatility confirmation
# Long when price breaks above Donchian(20) high + volume > 1.5x avg + ATR(14) rising
# Short when price breaks below Donchian(20) low + volume > 1.5x avg + ATR(14) rising
# Exit when price crosses opposite Donchian(10) level or ATR drops below threshold
# Uses price channel breakouts for trend following, volume for conviction, volatility for regime
# Designed to work in both bull and bear markets by capturing breakouts with filters
# Target: 50-100 total trades over 4 years (12-25/year) with size 0.25

name = "4h_Donchian_Breakout_Volume_Volatility"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    donchian_high_10 = high_series.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    # Volatility/ATR for trend confirmation and regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_rising = atr > np.roll(atr, 1)  # ATR increasing
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(vol_confirm[i]) or np.isnan(atr_rising[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian(20) high, volume spike, ATR rising
            if (close[i] > donchian_high_20[i] and 
                vol_confirm[i] and 
                atr_rising[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian(20) low, volume spike, ATR rising
            elif (close[i] < donchian_low_20[i] and 
                  vol_confirm[i] and 
                  atr_rising[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian(10) low or ATR drops
            if (close[i] < donchian_low_10[i]) or (not atr_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian(10) high or ATR drops
            if (close[i] > donchian_high_10[i]) or (not atr_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals