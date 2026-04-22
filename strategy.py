#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation
# This strategy trades breakouts of the 20-period Donchian channel on the 6h timeframe
# with trend alignment from 12h EMA(50) and volume confirmation.
# It works in both bull and bear markets by following the trend direction on higher timeframe.
# Uses discrete position sizing (0.25) to balance return and minimize transaction costs.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian and EMA trend (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) for previous period
    # Upper = max(high_12h[-20:])
    # Lower = min(low_12h[-20:])
    upper_20 = np.full_like(high_12h, np.nan)
    lower_20 = np.full_like(low_12h, np.nan)
    for i in range(20, len(high_12h)):
        upper_20[i] = np.max(high_12h[i-20:i])
        lower_20[i] = np.min(low_12h[i-20:i])
    
    # Align Donchian levels to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to avoid index issues
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above upper Donchian + above 12h EMA + volume spike
            if close[i] > upper_20_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Donchian + below 12h EMA + volume spike
            elif close[i] < lower_20_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 12h EMA in opposite direction
            if position == 1:
                # Exit long: Close below 12h EMA
                if close[i] < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Close above 12h EMA
                if close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hEMA50_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0