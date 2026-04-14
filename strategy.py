#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 12h Trend Filter and Volume Confirmation
# Long: Price breaks above Donchian(20) high with 12h EMA (50) uptrend and volume > 1.5x average
# Short: Price breaks below Donchian(20) low with 12h EMA (50) downtrend and volume > 1.5x average
# Exit: Price returns to Donchian midpoint or trend reverses
# Designed to work in both bull and bear markets by combining breakout momentum with trend filter
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA (50) for trend direction
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate volume average (20-period) for confirmation
    volume_series = pd.Series(volume)
    volume_avg = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for Donchian and volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_avg[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * volume_avg[i]
        
        if position == 0:
            # Long: breakout above Donchian high with uptrend and volume confirmation
            if price > donchian_high[i] and price > ema_12h_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Short: breakdown below Donchian low with downtrend and volume confirmation
            elif price < donchian_low[i] and price < ema_12h_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint or trend reverses
            if price <= donchian_mid[i] or price < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to midpoint or trend reverses
            if price >= donchian_mid[i] or price > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0