#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for higher timeframe trend alignment (stable in both bull/bear, less whipsaw than shorter HTF)
# Donchian(20) from prior 12h session provide clear breakout levels
# Volume confirmation (>2.0x 20 EMA) filters low-participation false breakouts
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 80-120 total trades over 4 years = 20-30/year for 12h.
# Works in both bull and bear: trend filter adapts to higher timeframe direction.

name = "12h_Donchian20_1dEMA50_VolumeSpike_Session"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian(20) channels from prior 12h bars
    # We need to calculate on 12h data, but we can use the current timeframe directly
    # since we're already on 12h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Donchian upper/lower bands (20-period)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0 and in_session:
            # Long conditions: price breaks above Donchian upper + uptrend + volume spike
            if close[i] > donchian_upper[i] and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + downtrend + volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend changes OR weak volume OR outside session
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2.0
            
            if (close[i] < donchian_mid or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i] or
                not in_session):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend changes OR weak volume OR outside session
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2.0
            
            if (close[i] > donchian_mid or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i] or
                not in_session):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals