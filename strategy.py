#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly EMA filter and volume confirmation.
# Long when price breaks above 20-day Donchian high AND price > weekly EMA(50) AND volume > 1.5x average.
# Short when price breaks below 20-day Donchian low AND price < weekly EMA(50) AND volume > 1.5x average.
# Exit when price returns to Donchian middle band.
# Uses Donchian channels for breakout signals, weekly EMA for trend filter, and volume for confirmation.
# Designed to work in both bull and bear markets by only trading in strong trends (price vs weekly EMA).
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for EMA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_middle[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian breakouts with weekly EMA filter
            # Long: price breaks above Donchian high AND price > weekly EMA(50) AND volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND price < weekly EMA(50) AND volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian middle
            if close[i] <= donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian middle
            if close[i] >= donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_WeeklyEMA_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0