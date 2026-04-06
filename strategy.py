#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 12-hour EMA trend filter and volume confirmation.
# Donchian(20) provides clear breakout signals in trending markets.
# EMA(50) on 12h timeframe filters for trend direction to avoid counter-trend trades.
# Volume confirmation ensures institutional participation.
# Designed for 4h timeframe to target 75-200 trades over 4 years with controlled frequency.

name = "4h_donchian20_12h_ema50_vol_v1"
timeframe = "4h"
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
    
    # Donchian channel (20-period) on 4h
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 12-hour EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA calculation
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_12h[49] = np.mean(close_12h[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema_12h[i] = (close_12h[i] - ema_12h[i-1]) * multiplier + ema_12h[i-1]
    
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 49)  # Donchian needs 19, EMA needs 49
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.2x average
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower band or stoploss
            if (close[i] < lowest_low[i] or 
                close[i] < entry_price - 2.5 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper band or stoploss
            if (close[i] > highest_high[i] or 
                close[i] > entry_price + 2.5 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume and trend filter
            if volume_filter:
                # Long: price breaks above upper band in uptrend (price > EMA)
                if close[i] > highest_high[i] and close[i] > ema_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below lower band in downtrend (price < EMA)
                elif close[i] < lowest_low[i] and close[i] < ema_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals