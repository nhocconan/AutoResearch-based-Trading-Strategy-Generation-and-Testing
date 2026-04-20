#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# - Long when price breaks above 20-day Donchian high AND price above 20-week EMA
# - Short when price breaks below 20-day Donchian low AND price below 20-week EMA
# - Require volume > 1.5x 20-day average volume for confirmation
# - Exit when price crosses 20-day EMA in opposite direction
# - Uses 1d timeframe with weekly trend filter to avoid counter-trend trades
# - Designed for low-frequency, high-quality signals to minimize fee drag
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-week EMA on weekly data
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily data for Donchian channels and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-day EMA for exit signal
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(avg_vol_20[i]) or \
           np.isnan(ema_20[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirmed = volume[i] > 1.5 * avg_vol_20[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above weekly EMA + volume
            if close[i] > high_20[i] and close[i] > ema_20_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below weekly EMA + volume
            elif close[i] < low_20[i] and close[i] < ema_20_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-day EMA
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-day EMA
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_WeeklyEMA_VolumeFilter"
timeframe = "1d"
leverage = 1.0