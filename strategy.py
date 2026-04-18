#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + 12h EMA Trend + Volume Spike
# Donchian(20) breakouts capture momentum in trending markets (both bull and bear).
# 12h EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume spike (>2x 12-period average) confirms institutional participation.
# Stops: exit when price closes back inside Donchian channel or reverses against EMA.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
name = "4h_Donchian20_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA34 on 12h data for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian(20) channels on 4h data (lookback window)
    # Using rolling window with min_periods to avoid look-ahead
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 12-period average volume (2 days on 4h chart)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > (2.0 * vol_ma_12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_12[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_max_val = high_max[i]
        low_min_val = low_min[i]
        ema_val = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: Close above upper Donchian AND price above 12h EMA AND volume spike
            if close_val > high_max_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Donchian AND price below 12h EMA AND volume spike
            elif close_val < low_min_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below lower Donchian (breakdown) or below EMA (trend change)
            if close_val < low_min_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above upper Donchian (breakout) or above EMA (trend change)
            if close_val > high_max_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals