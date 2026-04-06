#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA Trend Filter + Volume Confirmation
Hypothesis: Donchian breakouts capture momentum; 1d EMA filter ensures trades align with daily trend, reducing whipsaw in both bull and bear markets. Volume confirms breakout strength. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1dema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1h data for EMA trend filter (once before loop)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema_1h = np.full(len(close_1h), np.nan)
    if len(close_1h) >= 20:
        ema_1h[19] = np.mean(close_1h[:20])
        for i in range(20, len(close_1h)):
            ema_1h[i] = (close_1h[i] * 2 + ema_1h[i-1] * 19) / 21
    ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[0] = np.nan
            if len(tr) > 1:
                atr[1] = tr[0]
                for i in range(2, n):
                    atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ema_1h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + 1h EMA trend filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # 1h EMA trend filter: only long if price > EMA, short if price < EMA
            price_above_ema = close[i] > ema_1h_aligned[i]
            price_below_ema = close[i] < ema_1h_aligned[i]
            
            if bull_breakout and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals