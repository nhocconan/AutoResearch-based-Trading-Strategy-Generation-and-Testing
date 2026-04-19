#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, 1d EMA50 is rising, and volume > 1.5x 12h average volume.
# Short when price breaks below Donchian(20) low, 1d EMA50 is falling, and volume > 1.5x 12h average volume.
# Exit when price crosses the opposite Donchian band (20-period low for long, high for short).
# Uses Donchian for trend capture, EMA for trend filter, volume for confirmation.
# Target: 20-50 trades/year per symbol to stay within frequency limits.
name = "12h_Donchian20_EMA50_Volume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = np.nan
    ema_50_rising = ema_50_1d > ema_50_1d_prev
    ema_50_falling = ema_50_1d < ema_50_1d_prev
    
    # Align 1d EMA50 trend to 12h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    # Calculate Donchian channels (20-period) on 12h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ema_50_rising_val = ema_50_rising_aligned[i]
        ema_50_falling_val = ema_50_falling_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above Donchian high, 1d EMA50 rising, volume confirmed
            if price > high_20_val and ema_50_rising_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low, 1d EMA50 falling, volume confirmed
            elif price < low_20_val and ema_50_falling_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian low
            if price < low_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high
            if price > high_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals