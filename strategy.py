#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day trend filter (EMA50) and volume confirmation.
# Donchian breakouts capture momentum in trending markets.
# EMA50 filter ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation ensures institutional participation.
# Designed for 12h timeframe to target 50-150 trades over 4 years with low frequency.
# Works in both bull and bear markets by following the daily trend.

name = "12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1-day volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # Donchian needs 20 periods
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channels (20-period)
        highest_high = np.max(high[i-20:i+1])
        lowest_low = np.min(low[i-20:i+1])
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_avg_aligned[i] * 1.5
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or stoploss
            if (close[i] < lowest_low or 
                close[i] < entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or stoploss
            if (close[i] > highest_high or 
                close[i] > entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout in direction of daily trend
            if volume_filter:
                # Long: breakout above Donchian high in uptrend
                if close[i] > highest_high and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below Donchian low in downtrend
                elif close[i] < lowest_low and downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals