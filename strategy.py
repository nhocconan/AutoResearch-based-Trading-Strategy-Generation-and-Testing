#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Enter long when price breaks above Donchian(20) high, EMA50 is rising, volume > 1.5x average
# Enter short when price breaks below Donchian(20) low, EMA50 is falling, volume > 1.5x average
# Exit when price returns to Donchian midpoint or opposite breakout occurs
# Target: 40-80 trades over 4 years by combining multiple filters

name = "1d_donchian20_1w_ema50_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1w EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_prev = np.roll(ema_50, 1)
    ema_50_prev[0] = ema_50[0]  # Handle first value
    ema_50_rising = ema_50 > ema_50_prev
    ema_50_falling = ema_50 < ema_50_prev
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising.astype(float))
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to initialize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price below Donchian midpoint OR opposite breakout
            if close[i] < donchian_mid[i] or low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian midpoint OR opposite breakout
            if close[i] > donchian_mid[i] or high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA trend + volume
            if volume[i] > volume_threshold[i]:
                if (high[i] > donchian_high[i] and ema_50_rising_aligned[i] > 0.5):
                    # Bullish breakout with rising EMA50
                    signals[i] = 0.25
                    position = 1
                elif (low[i] < donchian_low[i] and ema_50_falling_aligned[i] > 0.5):
                    # Bearish breakout with falling EMA50
                    signals[i] = -0.25
                    position = -1
    
    return signals