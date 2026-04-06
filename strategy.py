#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h trend filter and volume confirmation.
# Uses Donchian(20) on 6h for breakout signals, 12h EMA50 for trend direction,
# and volume spike filter to ensure institutional participation.
# Works in bull markets (breakouts with trend) and bear markets (fade failed breakouts).
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "6h_donchian20_12h_ema50_vol_v1"
timeframe = "6h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Donchian channels on 6h (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume spike (2x average)
        vol_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian lower or trend turns bearish
            if close[i] < low_min[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian upper or trend turns bullish
            if close[i] > high_max[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter and trend alignment
            if vol_filter:
                # Bullish breakout: price above Donchian upper with bullish trend
                if close[i] > high_max[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below Donchian lower with bearish trend
                elif close[i] < low_min[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                # Fade failed breakouts: long at support in uptrend, short at resistance in downtrend
                elif close[i] <= low_min[i] and close[i] > ema_50_aligned[i]:
                    # Price at support but trend still bullish -> long
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= high_max[i] and close[i] < ema_50_aligned[i]:
                    # Price at resistance but trend still bearish -> short
                    signals[i] = -0.25
                    position = -1
    
    return signals