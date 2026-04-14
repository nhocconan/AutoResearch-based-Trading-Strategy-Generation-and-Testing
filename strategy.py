#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakouts with 12h EMA trend filter and volume confirmation.
# Long when price breaks above 12h Donchian upper with price above 12h EMA(20) and volume > 1.5x average.
# Short when price breaks below 12h Donchian lower with price below 12h EMA(20) and volume > 1.5x average.
# Exit when price returns to 12h Donchian midpoint or crosses 12h EMA.
# Uses 12h Donchian for structure, 12h EMA for trend filter, volume for confirmation.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Donchian channels and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Calculate 12h EMA(20)
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to lower timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 12h EMA
        price_above_ema = close[i] > ema_12h_aligned[i]
        price_below_ema = close[i] < ema_12h_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above Donchian upper AND price above 12h EMA
            if (close[i] > donchian_upper_aligned[i] and 
                price_above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian lower AND price below 12h EMA
            elif (close[i] < donchian_lower_aligned[i] and 
                  price_below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or price crosses below 12h EMA
            if (close[i] <= donchian_mid_aligned[i] or 
                close[i] < ema_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or price crosses above 12h EMA
            if (close[i] >= donchian_mid_aligned[i] or 
                close[i] > ema_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12hDonchian_12hEMA_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0