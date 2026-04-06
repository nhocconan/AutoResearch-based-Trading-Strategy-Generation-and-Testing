#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Daily EMA200 Trend + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum, EMA200 defines trend bias, volume confirms breakout strength, ATR stoploss limits drawdown.
Designed for low trade frequency (target 75-200 total over 4 years) to minimize fee decay. Works in both bull and bear by using EMA200 as trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_ata, align_htf_to_ltf

name = "6h_donchian20_ema200_vol_atr_v1"
timeframe = "6h"
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
    
    # Load daily data once (HTF)
    df_1d = get_htf_data(prices, '1d')
    # Calculate EMA200 on daily close
    close_1d = df_1d['close'].values
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = close_1d[:200].mean()
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 + ema_200_1d[i-1] * 198) / 200
    # Align to 6h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ema_200_1d_aligned[i]):
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
            # Look for entries: Donchian breakout + volume + EMA200 trend filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # EMA200 trend filter: only long if price > EMA200, short if price < EMA200
            price_above_ema = close[i] > ema_200_1d_aligned[i]
            price_below_ema = close[i] < ema_200_1d_aligned[i]
            
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
</pre>