#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour 4-hour Donchian breakout with daily EMA50 filter and volume confirmation
# Uses 4h for signal direction and 1h for entry timing to reduce trade frequency
# Long when price breaks above 4h Donchian(20) high AND price > daily EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below 4h Donchian(20) low AND price < daily EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back inside the 4h Donchian channel (opposite band)
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    # Load daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_4h_aligned[i]) or np.isnan(low_20_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above 4h Donchian high + above daily EMA50 + volume confirmation
            if (price > high_20_4h_aligned[i] and price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below 4h Donchian low + below daily EMA50 + volume confirmation
            elif (price < low_20_4h_aligned[i] and price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below 4h Donchian low (opposite band)
            if price < low_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above 4h Donchian high (opposite band)
            if price > high_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hDonchian_1dEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0