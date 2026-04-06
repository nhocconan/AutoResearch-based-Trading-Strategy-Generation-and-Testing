#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high AND weekly EMA(10) rising AND volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low AND weekly EMA(10) falling AND volume > 1.5x avg
# Exit when: price crosses opposite Donchian band OR volume drops below average
# Targets 40-80 trades over 4 years (10-20/year) for low fee drag and robust performance

name = "1d_donchian20_weeklyema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA(10) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_10 = pd.Series(close_1w).ewm(span=10, adjust=False).mean().values
    ema_10_prev = np.roll(ema_10, 1)
    ema_10_prev[0] = ema_10[0]  # handle first value
    ema_10_rising = ema_10 > ema_10_prev
    ema_10_falling = ema_10 < ema_10_prev
    ema_10_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_10_rising)
    ema_10_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_10_falling)
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(ema_10_rising_aligned[i]) or
            np.isnan(ema_10_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian low OR volume drops below average
            if close[i] < donchian_low[i] or volume[i] < volume_avg[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian high OR volume drops below average
            if close[i] > donchian_high[i] or volume[i] < volume_avg[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and ema_10_rising_aligned[i]:
                    # Bullish breakout with rising weekly trend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and ema_10_falling_aligned[i]:
                    # Bearish breakout with falling weekly trend
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>