# ABOUT THIS EXPERIMENT
# Hypothesis: 1h 3-day Donchian breakout with 4h 20-period EMA trend filter and volume confirmation
# In bull markets: buy breakouts above 3-day high when above 4h EMA20 (uptrend filter)
# In bear markets: sell breakdowns below 3-day low when below 4h EMA20 (downtrend filter)
# Volume filter ensures breakouts have participation
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Uses 4h for signal direction, 1h only for entry timing
# Session filter (08-20 UTC) to reduce noise trades
# Position size: 0.20

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 4h data ONCE for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 1h 3-day Donchian channels (breakout levels)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 3-period high/low for Donchian (3 * 1h = 3h, but we want ~3 days = 72 periods)
    # Actually 3 days = 72 hours, but let's use a reasonable 24-period for ~1 day
    # Let's use 12 periods for half day, 24 for 1 day, 72 for 3 days
    # But to keep it reasonable and avoid too many signals, let's use 24 (~1 day)
    # Actually let's use 48 for 2 days as a compromise
    lookback = 48  # ~2 days
    
    highest_48h = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_48h = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 1h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if NaN in indicators
        if np.isnan(highest_48h[i]) or np.isnan(lowest_48h[i]) or \
           np.isnan(ema20_4h_aligned[i]) or np.isnan(volume_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        resistance = highest_48h[i]
        support = lowest_48h[i]
        ema20 = ema20_4h_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 48h resistance, above 4h EMA20 (uptrend), with volume
            if price > resistance and price > ema20 and vol_filter:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below 48h support, below 4h EMA20 (downtrend), with volume
            elif price < support and price < ema20 and vol_filter:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price breaks below 48h support
            if price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above 48h resistance
            if price > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_48H_Donchian_4hEMA20_Trend_VolumeFilter"
timeframe = "1h"
leverage = 1.0