#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day EMA(50) trend filter and 1-week volume confirmation.
# Uses price channel breakouts for trend following in both bull and bear markets.
# EMA filter ensures trading only in the direction of higher timeframe trend.
# Volume confirmation ensures institutional participation.
# Designed for 12h timeframe to target 50-150 trades over 4 years with low frequency.

name = "12h_donchian20_1d_ema50_vol_v1"
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
        multiplier = 2 / (50 + 1)
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] - ema_1d[i-1]) * multiplier + ema_1d[i-1]
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    if len(vol_1w) >= 5:
        for i in range(4, len(vol_1w)):
            vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_period = 14
    
    # Pre-calculate ATR for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    atr = np.zeros(n)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[1:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Start from warmup period
    start = max(50, 4)  # EMA needs 50, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channels (20-period)
        if i >= 20:
            highest_high = np.max(high[i-19:i+1])
            lowest_low = np.min(low[i-19:i+1])
        else:
            highest_high = np.max(high[:i+1])
            lowest_low = np.min(low[:i+1])
        
        # Volume condition: current volume > 1.5x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below Donchian lower band or stoploss
            if (close[i] <= lowest_low or 
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian upper band or stoploss
            if (close[i] >= highest_high or 
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long: breakout above upper band in uptrend
                if close[i] > highest_high and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below lower band in downtrend
                elif close[i] < lowest_low and downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals