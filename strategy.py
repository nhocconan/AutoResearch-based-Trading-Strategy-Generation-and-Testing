#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter (EMA50) and volume confirmation.
# Long when price breaks above 6h Donchian(20) upper band AND 12h EMA50 is rising AND volume > 1.5x 20-period average.
# Short when price breaks below 6h Donchian(20) lower band AND 12h EMA50 is falling AND volume > 1.5x 20-period average.
# Uses volume breakout confirmation to avoid false breaks and EMA trend filter to align with higher timeframe momentum.
# Designed for moderate trade frequency (target: 20-40/year) to balance opportunity and cost.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.
name = "6h_Donchian20_Volume_EMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend direction
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_rising = ema50_12h > np.roll(ema50_12h, 1)  # rising if current > previous
    ema50_12h_falling = ema50_12h < np.roll(ema50_12h, 1)  # falling if current < previous
    ema50_12h_rising[0] = False  # first value has no previous
    ema50_12h_falling[0] = False
    
    ema50_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_rising)
    ema50_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_falling)
    
    # 6h Donchian(20) channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(ema50_12h_rising_aligned[i]) or np.isnan(ema50_12h_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above Donchian high, rising 12h EMA50, volume surge
            long_condition = (close[i] > donchian_high_20[i]) and ema50_12h_rising_aligned[i] and volume_filter[i]
            # Short condition: break below Donchian low, falling 12h EMA50, volume surge
            short_condition = (close[i] < donchian_low_20[i]) and ema50_12h_falling_aligned[i] and volume_filter[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or volume drops significantly
            if (close[i] < donchian_low_20[i]) or (volume[i] < (0.5 * vol_ma_20[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or volume drops significantly
            if (close[i] > donchian_high_20[i]) or (volume[i] < (0.5 * vol_ma_20[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals