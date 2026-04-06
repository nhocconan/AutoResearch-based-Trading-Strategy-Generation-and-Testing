#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout (20-period) with 12-hour EMA trend filter and volume confirmation.
# Go long when price breaks above Donchian upper band in uptrend (EMA20 rising) with above-average volume.
# Go short when price breaks below Donchian lower band in downtrend (EMA20 falling) with above-average volume.
# Uses volume filter to avoid false breakouts. Designed for 4h timeframe to target 75-200 trades over 4 years.
# Works in bull (captures breakouts) and bear (short breakdowns) markets.

name = "4h_donchian20_12h_ema20_vol_v1"
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
    
    # 4-hour Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):  # 20-period lookback
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 12-hour EMA(20) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 20:
        ema_12h[19] = np.mean(close_12h[:20])  # SMA seed
        for i in range(20, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 / (20 + 1)) + (ema_12h[i-1] * (19 / (20 + 1)))
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # EMA slope (trend direction)
    ema_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema_12h_aligned[i]) and not np.isnan(ema_12h_aligned[i-1]):
            ema_slope[i] = ema_12h_aligned[i] - ema_12h_aligned[i-1]
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):  # 20-period average
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 20, 19)  # Donchian needs 20, EMA needs 20, volume needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_slope[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.2x average
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower band or stoploss
            if (close[i] < lowest_low[i] or 
                close[i] < entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper band or stoploss
            if (close[i] > highest_high[i] or 
                close[i] > entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            if volume_filter:
                # Long: break above upper band in uptrend (EMA rising)
                if (close[i] > highest_high[i] and ema_slope[i] > 0):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: break below lower band in downtrend (EMA falling)
                elif (close[i] < lowest_low[i] and ema_slope[i] < 0):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals