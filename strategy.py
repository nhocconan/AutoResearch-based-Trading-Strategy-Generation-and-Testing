#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum breakout with 4h trend filter and volume confirmation
# Long when price breaks above 1h high of last 20 bars AND 4h EMA50 is rising AND volume > 1.5x 20-period average
# Short when price breaks below 1h low of last 20 bars AND 4h EMA50 is falling AND volume > 1.5x 20-period average
# Exit when price crosses back inside the 1h channel (opposite band)
# Uses 4h for trend direction, 1h for entry timing to reduce false signals
# Target: 100-150 total trades over 4 years (25-38/year) to balance opportunity and cost
# Session filter: 08-20 UTC to avoid low-volume periods

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1h Donchian channels (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h EMA50 for trend filter (rising/falling)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_rising = ema50_4h > np.roll(ema50_4h, 1)
    ema50_4h_falling = ema50_4h < np.roll(ema50_4h, 1)
    ema50_4h_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_rising.astype(float))
    ema50_4h_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_falling.astype(float))
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_4h_rising_aligned[i]) or np.isnan(ema50_4h_falling_aligned[i]) or 
            np.isnan(vol_avg[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above 1h high + 4h EMA50 rising + volume confirmation
            if (price > high_20[i] and ema50_4h_rising_aligned[i] > 0.5 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below 1h low + 4h EMA50 falling + volume confirmation
            elif (price < low_20[i] and ema50_4h_falling_aligned[i] > 0.5 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below 1h low (opposite band)
            if price < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above 1h high (opposite band)
            if price > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_Momentum_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0