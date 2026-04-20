#!/usr/bin/env python3
# 4h_12h_Donchian20_Breakout_VolumeTrend
# Hypothesis: On 4h timeframe, trade breakouts from 20-period Donchian channels with volume spike confirmation and 12h EMA trend filter.
# Designed to work in both bull and bear markets by following the 12h trend. Targets 20-40 trades per year.
# Breakouts require price > upper band + 0.5% buffer (long) or < lower band - 0.5% buffer (short).
# Volume confirmation: current volume > 2x 20-period average. Exit on trend reversal (price crosses 12h EMA).

name = "4h_12h_Donchian20_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period Donchian channels on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_12h_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Donchian upper + buffer, volume spike, and price above 12h EMA34 (uptrend)
            if (close[i] > donchian_upper[i] * 1.005 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below Donchian lower - buffer, volume spike, and price below 12h EMA34 (downtrend)
            elif (close[i] < donchian_lower[i] * 0.995 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below 12h EMA34 (trend reversal)
            if close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above 12h EMA34 (trend reversal)
            if close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals