#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1w EMA Trend + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts on 12h capture medium-term momentum, 1w EMA ensures trend alignment with weekly bias, volume confirms breakout strength. Designed for low trade frequency (target 50-150 total over 4 years) with wide stops to avoid whipsaws. Works in bull/bear by only trading with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1wema_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for EMA and ATR (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 50-period EMA on weekly
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        multiplier = 2 / (50 + 1)
        ema_1w[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # 14-period ATR on weekly for stoploss
    atr_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 14:
        tr = np.maximum(
            high_1w[1:] - low_1w[1:],
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
        atr_1w[0] = np.nan
        if len(tr) > 0:
            atr_1w[1] = tr[0]
            for i in range(2, len(atr_1w)):
                atr_1w[i] = (tr[i-1] * 13 + atr_1w[i-1]) / 14
    
    # Align indicators to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 50)  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below weekly EMA
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or 
                close[i] < ema_1w_aligned[i] or
                close[i] < entry_price - 2.0 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above weekly EMA
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or 
                close[i] > ema_1w_aligned[i] or
                close[i] > entry_price + 2.0 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA trend filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Only go long if price above weekly EMA, short if below
            trend_uptrend = close[i] > ema_1w_aligned[i]
            trend_downtrend = close[i] < ema_1w_aligned[i]
            
            if i >= 20 and bull_breakout and volume_filter and trend_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif i >= 20 and bear_breakout and volume_filter and trend_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals