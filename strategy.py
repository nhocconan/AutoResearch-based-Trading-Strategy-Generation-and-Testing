#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout (20-period) with 1-day EMA trend filter and volume confirmation.
# Donchian breakouts capture momentum bursts in both bull and bear markets.
# EMA filter ensures trades align with higher timeframe trend to avoid counter-trend whipsaws.
# Volume confirmation filters for institutional participation.
# Designed for 12h timeframe to target 50-150 trades over 4 years with low frequency.

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
    
    # EMA calculation
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_1d[i-1] * (49 / (50 + 1)))
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
    start = 49  # EMA needs 50 periods
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Trend condition: price above/below EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Donchian channel breakout (20-period)
        if i >= 19:  # Need 20 periods for lookback
            highest_high = np.max(high[i-19:i+1])
            lowest_low = np.min(low[i-19:i+1])
            
            # Check exits and stoploss
            if position == 1:  # long position
                # Exit: price breaks below Donchian low or stoploss
                if (close[i] < lowest_low or 
                    close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:  # short position
                # Exit: price breaks above Donchian high or stoploss
                if (close[i] > highest_high or 
                    close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Look for entries: breakout with trend and volume
                if volume_filter:
                    # Long: break above Donchian high in uptrend
                    if close[i] > highest_high and above_ema:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Short: break below Donchian low in downtrend
                    elif close[i] < lowest_low and below_ema:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals