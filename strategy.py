#!/usr/bin/env python3
# 4h_1d_Donchian_Breakout_VolumeTrend_Regime
# Hypothesis: On 4h timeframe, trade Donchian(20) breakouts with 1d trend filter (EMA50) and volume confirmation.
# In bull markets (price > EMA50), take long breakouts; in bear markets (price < EMA50), take short breakdowns.
# Volume must exceed 1.5x 20-period average to confirm breakout strength.
# Targets 20-40 trades/year by requiring trend alignment and volume confirmation.
# Works in both bull and bear: trend filter ensures we only trade in direction of higher timeframe trend.

name = "4h_1d_Donchian_Breakout_VolumeTrend_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend direction from 1d EMA50
            uptrend = close[i] > ema_50_aligned[i]
            downtrend = close[i] < ema_50_aligned[i]
            
            # Long breakout: price above Donchian upper band in uptrend with volume
            if uptrend and close[i] > highest_high[i]:
                if volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price below Donchian lower band in downtrend with volume
            elif downtrend and close[i] < lowest_low[i]:
                if volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian middle or trend reverses
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < donchian_mid or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian middle or trend reverses
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > donchian_mid or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals