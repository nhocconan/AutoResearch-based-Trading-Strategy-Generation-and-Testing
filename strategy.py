#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Volume Filter + ATR Stoploss + 1w Trend Filter
Hypothesis: On daily timeframe, Donchian breakouts with volume confirmation and weekly trend filter capture strong trends while minimizing whipsaw. Weekly trend ensures alignment with higher timeframe momentum, reducing trades during counter-trend moves. Designed for low trade frequency (target 30-100 total over 4 years) to minimize fee decay.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_vol_atr_1wtrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for indicators
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
    
    # Get weekly trend data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) > 0:
        # Weekly EMA(21) for trend
        close_1w = df_1w['close'].values
        ema_1w = np.full(len(close_1w), np.nan)
        if len(close_1w) >= 21:
            ema_multiplier = 2 / (21 + 1)
            ema_1w[20] = np.mean(close_1w[:21])
            for i in range(21, len(close_1w)):
                ema_1w[i] = (close_1w[i] * ema_multiplier) + (ema_1w[i-1] * (1 - ema_multiplier))
        # Align weekly EMA to daily timeframe
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    else:
        ema_1w_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian
    
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
        
        # Weekly trend filter: price above/below weekly EMA
        price_above_weekly_ema = close[i] > ema_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_1w_aligned[i]
        
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
            
            # Only take longs in uptrend (price above weekly EMA) and shorts in downtrend
            if bull_breakout and volume_filter and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals