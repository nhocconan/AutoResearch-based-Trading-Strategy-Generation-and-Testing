#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Uses 12h EMA50 for trend direction to avoid counter-trend entries.
# Breakouts require volume > 1.5x 20-period EMA for institutional confirmation.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
name = "6h_Donchian20_12hEMA50_Trend_Volume"
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
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6-period high/low for Donchian channels (20 periods on 6h = 5 days)
    # Using 20-period lookback for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 20-period high with volume spike and above 12h EMA50
            if (price > high_20[i] and vol_spike[i] and price > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with volume spike and below 12h EMA50
            elif (price < low_20[i] and vol_spike[i] and price < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-period low (mean reversion)
            if price < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-period high (mean reversion)
            if price > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals