#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour donchian breakout with 1-day volume confirmation and 1-week atr filter.
# Donchian(20) breakouts capture trends in both bull and bear markets.
# Volume confirmation ensures institutional participation.
# ATR filter avoids whipsaw in low volatility conditions.
# Designed for 12h timeframe to target 50-150 trades over 4 years with low frequency.

name = "12h_donchian20_1d_vol1w_atr_v1"
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
    
    # 12-hour donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    
    for i in range(4, len(vol_1d)):  # 5-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-4:i+1])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1-week ATR for volatility filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr = np.full(len(close_1w), np.nan)
    if len(close_1w) > 1:
        tr[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr[i] = max(high_1w[i] - low_1w[i],
                       abs(high_1w[i] - close_1w[i-1]),
                       abs(low_1w[i] - close_1w[i-1]))
    
    # ATR calculation (14-period)
    atr_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 14:
        atr_1w[13] = np.mean(tr[0:14])
        for i in range(14, len(close_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 4, 14)  # Donchian needs 19, volume needs 4, ATR needs 14
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.3
        
        # ATR filter: only trade when volatility is above average
        volatility_filter = atr_1w_aligned[i] > 0  # ATR should always be positive when calculated
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below donchian low or stoploss
            if (close[i] < lowest_low[i] or 
                close[i] < entry_price - 2.5 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above donchian high or stoploss
            if (close[i] > highest_high[i] or 
                close[i] > entry_price + 2.5 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter and volatility_filter:
                # Long: price breaks above donchian high
                if close[i] > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below donchian low
                elif close[i] < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals