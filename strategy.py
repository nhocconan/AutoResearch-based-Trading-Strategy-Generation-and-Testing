#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly EMA(20) trend filter and volume confirmation.
# Enter long when price breaks above Donchian upper with weekly EMA(20) rising and volume > 1.5x avg.
# Enter short when price breaks below Donchian lower with weekly EMA(20) falling and volume > 1.5x avg.
# Exit on opposite Donchian breakout or when price crosses weekly EMA(20).
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.
# Weekly EMA provides strong trend filter to avoid whipsaws in both bull and bear markets.

name = "12h_donchian20_weekly_ema20_vol_v1"
timeframe = "12h"
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
    
    # Weekly EMA(20) for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    ema_20 = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_weekly, ema_20)
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR crosses below EMA20
            if close[i] < donchian_low[i] or close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR crosses above EMA20
            if close[i] > donchian_high[i] or close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA20 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and close[i] > ema_20_aligned[i]:
                    # Breakout above Donchian high in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and close[i] < ema_20_aligned[i]:
                    # Breakdown below Donchian low in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals