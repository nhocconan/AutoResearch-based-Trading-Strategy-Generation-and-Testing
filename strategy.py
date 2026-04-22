#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d pivot direction filter and volume confirmation
# Donchian breakouts capture momentum in trending markets.
# 1d pivot points (PP) provide institutional support/resistance levels:
#   Breakout above PP = bullish bias, breakdown below PP = bearish bias.
# Volume > 1.5x average confirms institutional participation.
# Works in both bull/bear by following price action relative to daily pivot.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for pivot points (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d pivot point: PP = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Donchian channels on 6h data
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donchian_period, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + above 1d PP + volume spike
            if (close[i] > upper[i] and 
                close[i] > pp_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + below 1d PP + volume spike
            elif (close[i] < lower[i] and 
                  close[i] < pp_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian band or crosses 1d PP
            if position == 1:
                # Exit long: Price below lower Donchian or below 1d PP
                if close[i] < lower[i] or close[i] < pp_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price above upper Donchian or above 1d PP
                if close[i] > upper[i] or close[i] > pp_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dPP_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0