#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA(50) trend filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets, while EMA(50) filters for trend direction.
# Volume confirmation ensures institutional participation, reducing false breakouts.
# Designed for 4h timeframe to target 75-200 trades over 4 years with controlled frequency.

name = "4h_donchian20_1d_ema50_vol_v1"
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
    
    # 1-day EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA(50) calculation
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2/51) + (ema_1d[i-1] * 49/51)
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-day volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian(20) channels
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(49, 19)  # EMA needs 49, Donchian needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.3
        
        # Trend filter: price above/below EMA(50)
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below Donchian lower band or stoploss
            if (close[i] < lowest_low[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian upper band or stoploss
            if (close[i] > highest_high[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakouts with volume and trend alignment
            if volume_filter:
                # Long: breakout above upper band in uptrend
                if (close[i] > highest_high[i] and uptrend):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below lower band in downtrend
                elif (close[i] < lowest_low[i] and downtrend):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals