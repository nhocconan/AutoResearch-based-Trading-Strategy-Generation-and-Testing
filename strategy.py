#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with weekly trend filter and volume confirmation.
# Enter long when price breaks above Donchian(20) upper band with weekly EMA(10) rising and volume > 2x avg.
# Enter short when price breaks below Donchian(20) lower band with weekly EMA(10) falling and volume > 2x avg.
# Exit on opposite Donchian breakout or when price crosses weekly EMA(10).
# Target: 30-100 total trades over 4 years (7-25/year) with controlled risk.

name = "1d_donchian20_weeklyema10_vol_v1"
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
    
    # Weekly EMA(10) for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    ema_10 = pd.Series(close_weekly).ewm(span=10, adjust=False).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_weekly, ema_10)
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_10_aligned[i]) or np.isnan(volume_threshold[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR crosses below weekly EMA10
            if close[i] < donchian_lower[i] or close[i] < ema_10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR crosses above weekly EMA10
            if close[i] > donchian_upper[i] or close[i] > ema_10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly EMA10 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_upper[i] and close[i] > ema_10_aligned[i]:
                    # Breakout above Donchian upper in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_lower[i] and close[i] < ema_10_aligned[i]:
                    # Breakdown below Donchian lower in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals