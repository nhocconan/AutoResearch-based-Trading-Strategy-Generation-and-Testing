#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Enter long when price breaks above Donchian upper band AND price > 1d EMA50 AND volume > 1.5x average
# Enter short when price breaks below Donchian lower band AND price < 1d EMA50 AND volume > 1.5x average
# Exit when price crosses the opposite Donchian band OR trend filter reverses OR volume drops below threshold
# Uses 4h timeframe to target 75-200 trades over 4 years with clear entry/exit logic

name = "4h_donchian20_1dema_volume_v1"
timeframe = "4h"
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
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below lower Donchian band OR trend reverses OR low volume
            if close[i] < donchian_lower[i] or close[i] < ema_50_aligned[i] or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above upper Donchian band OR trend reverses OR low volume
            if close[i] > donchian_upper[i] or close[i] > ema_50_aligned[i] or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume confirmation
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_upper[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above resistance with uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_lower[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakdown below support with downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals