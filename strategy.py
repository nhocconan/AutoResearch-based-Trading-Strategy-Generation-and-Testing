#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Long when price breaks above 1d Donchian(20) high AND 1w EMA(10) is rising AND volume above average.
# Short when price breaks below 1d Donchian(20) low AND 1w EMA(10) is falling AND volume above average.
# Exit when price crosses opposite Donchian boundary or reverses against 1w EMA.
# Works in bull markets via breakout continuation and bear markets via mean reversion at extremes.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_donchian20_1w_ema10_vol_v1"
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
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) - using previous day's data to avoid look-ahead
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1w EMA(10) for trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_1w_rising = pd.Series(ema_1w).diff() > 0
    ema_1w_falling = pd.Series(ema_1w).diff() < 0
    
    # Align indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_rising.values.astype(float))
    ema_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_falling.values.astype(float))
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1w_rising_aligned[i]) or np.isnan(ema_1w_falling_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian low OR 1w EMA turns down
            if close[i] <= donchian_low_aligned[i] or not ema_1w_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian high OR 1w EMA turns up
            if close[i] >= donchian_high_aligned[i] or not ema_1w_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if vol_filter:
                # Long breakout: price above Donchian high AND 1w EMA rising
                if close[i] > donchian_high_aligned[i] and ema_1w_rising_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below Donchian low AND 1w EMA falling
                elif close[i] < donchian_low_aligned[i] and ema_1w_falling_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals