#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 12h EMA(50) rising AND volume > 1.5x average.
# Short when price breaks below Donchian(20) low AND 12h EMA(50) falling AND volume > 1.5x average.
# Exit when price crosses Donchian midpoint or 12h EMA reverses.
# This captures breakouts with trend alignment and volume confirmation to reduce false signals.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    midpoint = (highest_high + lowest_low) / 2.0
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data ONCE for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA(50) on 12h data
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.roll(ema_50, 1)
    ema_50_prev[0] = np.nan
    ema_rising = ema_50 > ema_50_prev
    ema_falling = ema_50 < ema_50_prev
    
    # Align 12h indicators to 4h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(avg_volume[i]) or
            np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high with rising 12h EMA and volume confirmation
            if (close[i] > highest_high[i] and 
                ema_rising_aligned[i] and
                volume[i] > 1.5 * avg_volume[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below Donchian low with falling 12h EMA and volume confirmation
            elif (close[i] < lowest_low[i] and 
                  ema_falling_aligned[i] and
                  volume[i] > 1.5 * avg_volume[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses Donchian midpoint or EMA falls
            if (close[i] < midpoint[i] or 
                not ema_rising_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses Donchian midpoint or EMA rises
            if (close[i] > midpoint[i] or 
                not ema_falling_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_12hEMA_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0