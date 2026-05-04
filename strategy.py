#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume confirmation
# In trending markets (12h EMA50 slope > 0), we trade breakouts: long on upper Donchian breakout, short on lower.
# Volume confirmation (>1.3x 20-period EMA) reduces false breakouts. Designed for 4h timeframe targeting 75-200 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "4h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h EMA50 slope (trend direction)
    ema_slope = np.zeros_like(ema_50_12h_aligned)
    ema_slope[1:] = ema_50_12h_aligned[1:] - ema_50_12h_aligned[:-1]
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = np.zeros(n)
    donchian_lower = np.zeros(n)
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Only trade in direction of 12h EMA50 trend
            if ema_slope[i] > 0 and volume_confirm:  # Uptrend
                if close[i] > donchian_upper[i]:
                    signals[i] = 0.25
                    position = 1
            elif ema_slope[i] < 0 and volume_confirm:  # Downtrend
                if close[i] < donchian_lower[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches Donchian midpoint OR EMA trend weakens
            midpoint = (donchian_upper[i] + donchian_lower[i]) / 2
            if (close[i] <= midpoint or 
                ema_slope[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Donchian midpoint OR EMA trend weakens
            midpoint = (donchian_upper[i] + donchian_lower[i]) / 2
            if (close[i] >= midpoint or 
                ema_slope[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals