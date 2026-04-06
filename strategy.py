#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + 12h EMA Trend Filter + ATR Stoploss
Hypothesis: Donchian breakouts with volume spike (>2x average) and strong trend (12h EMA > 12h EMA 5 periods ago) capture high-probability moves. 12h EMA filter prevents whipsaws in ranging markets. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_vol_12hema_v1"
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
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 12h EMA for trend filter (using mtf_data)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) > 0:
        close_12h = df_12h['close'].values
        ema_12h = np.full(len(close_12h), np.nan)
        if len(close_12h) >= 21:
            ema_12h[20] = np.mean(close_12h[:21])
            for i in range(21, len(close_12h)):
                ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 19) / 21
        ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    else:
        ema_12h_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 25  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ema_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 2.0  # Increased threshold for fewer trades
        
        # 12h EMA trend filter (trending up)
        trend_filter = ema_12h_aligned[i] > ema_12h_aligned[i-5] if i >= 5 else False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + 12h EMA trend filter
            # Minimum holding period: only allow new entry after 25 bars flat
            if bars_since_entry >= 25:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                if bull_breakout and volume_filter and trend_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and not trend_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals