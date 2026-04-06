#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Volume Filter + ATR Stoploss with 1w EMA Trend Filter
Hypothesis: Daily timeframe reduces trade frequency while capturing major trends. 
Weekly EMA filter ensures we only trade with the dominant trend, reducing whipsaw.
Volume confirmation ensures breakout strength. ATR stoploss limits drawdown.
Designed for low trade frequency (target 30-100 total over 4 years) to minimize fee decay.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_vol_atr_1wtrend_v2"
timeframe = "1d"
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
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    # 50-period EMA on weekly
    ema_1w = np.full(len(weekly_close), np.nan)
    if len(weekly_close) >= 50:
        ema_1w[49] = np.mean(weekly_close[:50])
        for i in range(50, len(weekly_close)):
            ema_1w[i] = (weekly_close[i] * 2 + ema_1w[i-1] * 49) / 50
    
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ema_1w_aligned[i]):
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
            # Look for entries: Donchian breakout + volume + weekly trend filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Weekly trend filter: only long above weekly EMA, short below
            above_weekly_ema = close[i] > ema_1w_aligned[i]
            below_weekly_ema = close[i] < ema_1w_aligned[i]
            
            if bull_breakout and volume_filter and above_weekly_ema:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and below_weekly_ema:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals