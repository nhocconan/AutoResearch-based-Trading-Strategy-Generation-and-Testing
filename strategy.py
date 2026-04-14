#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian Breakout with 4h Trend Filter and Volume Confirmation
# Uses 1h Donchian channel breakouts (20-period) for entries
# 4h EMA (50) provides trend filter to avoid counter-trend trades
# Volume > 1.5x average confirms breakout strength
# Timeframe: 1h (primary) with 4h trend filter
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag
# Works in both bull and bear markets by following the trend with volume confirmation

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
    
    # Calculate 4h EMA (50) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period)
    volume_series = pd.Series(volume)
    volume_avg = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 20  # for Donchian and volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_avg[i]) or np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 4h EMA
        above_ema = price > ema_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and uptrend
            if price > donchian_high[i] and vol > 1.5 * volume_avg[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume confirmation and downtrend
            elif price < donchian_low[i] and vol > 1.5 * volume_avg[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend changes
            if price < donchian_low[i] or price < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend changes
            if price > donchian_high[i] or price > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_Donchian_Breakout_4hEMA_Volume"
timeframe = "1h"
leverage = 1.0