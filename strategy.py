#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-week ATR filter and 1-day volume confirmation.
# Breakouts capture momentum in trending markets, while ATR filter avoids low-volatility false breakouts.
# Volume confirmation ensures institutional participation. Designed for 12h to target 50-150 trades over 4 years.
# Works in both bull (breakouts continue) and bear (breakdowns) markets.

name = "12h_donchian20_1w_atr1d_vol_v1"
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
    
    # 1-day Donchian(20) for breakout levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels
    highest_high = np.full(len(high_1d), np.nan)
    lowest_low = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):  # 20-period lookback
        highest_high[i] = np.max(high_1d[i-19:i+1])
        lowest_low[i] = np.min(low_1d[i-19:i+1])
    
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # 1-week ATR(14) for volatility filter
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
    
    # ATR calculation
    atr_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 14:
        atr_1w[13] = np.nanmean(tr[0:14])
        for i in range(14, len(close_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 1-day volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 14, 20)  # Donchian needs 20, ATR needs 14, volume needs 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # ATR filter: only trade when volatility is sufficient (ATR > 0.5 * 20-period average)
        atr_ma = np.full(len(close_1w), np.nan)
        if len(close_1w) >= 34:  # Need enough data for 20-period ATR average
            atr_ma[33] = np.nanmean(atr_1w[20:34])  # 14-period average of ATR
            for j in range(34, len(close_1w)):
                atr_ma[j] = (atr_ma[j-1] * 19 + atr_1w[j]) / 20
        
        atr_ma_aligned = align_htf_to_ltf(prices, df_1w, atr_ma)
        volatility_filter = atr_1w_aligned[i] > 0.5 * atr_ma_aligned[i] if not np.isnan(atr_ma_aligned[i]) else True
        
        # Volume condition: current volume > 1.2x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.2
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price retouches lower Donchian band or stoploss
            if (close[i] <= lowest_low_aligned[i] or 
                close[i] < entry_price - 2.5 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price retouches upper Donchian band or stoploss
            if (close[i] >= highest_high_aligned[i] or 
                close[i] > entry_price + 2.5 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            if volatility_filter and volume_filter:
                # Long breakout: price breaks above upper Donchian band
                if close[i] > highest_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below lower Donchian band
                elif close[i] < lowest_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals