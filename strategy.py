#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Daily VWAP Filter + Volume Spike
Hypothesis: Donchian breakouts with volume confirmation and price above/below daily VWAP 
capture institutional moves. VWAP filter avoids false breakouts in range-bound markets. 
Target: 80-150 total trades over 4 years (20-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_vwap_vol_v1"
timeframe = "6h"
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
    
    # Get daily VWAP (HTF)
    df_1d = get_htf_data(prices, '1d')
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_raw = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_raw.values
    vwap_6h = align_htf_to_ltf(prices, df_1d, vwap_1d)  # includes shift(1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(vwap_6h[i]):
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
        volume_filter = volume[i] > vol_ma * 2.0
        
        # VWAP filter
        price_above_vwap = close[i] > vwap_6h[i]
        price_below_vwap = close[i] < vwap_6h[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower or VWAP
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < vwap_6h[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper or VWAP
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > vwap_6h[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + VWAP filter
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high and price_above_vwap
                bear_breakout = close[i] < lowest_low and price_below_vwap
                
                if bull_breakout and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter:
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