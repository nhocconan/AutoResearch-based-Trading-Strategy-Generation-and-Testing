#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h Supertrend(ATR=10,mult=3) + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts on 4h capture momentum aligned with 12h Supertrend, volume confirms breakout strength, ATR stoploss limits drawdown. Targeting 75-200 total trades over 4 years with strict entry criteria.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12hsupertrend_vol_v1"
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
    
    # Load 12h data for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h ATR (10-period)
    atr_12h = np.full(len(high_12h), np.nan)
    if len(high_12h) >= 10:
        tr_12h = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
        if len(tr_12h) > 0:
            atr_12h[1] = tr_12h[0]
            for i in range(2, len(high_12h)):
                atr_12h[i] = (tr_12h[i-1] * 9 + atr_12h[i-1]) / 10
    
    # 12h Supertrend (ATR=10, mult=3)
    supertrend_12h = np.full(len(high_12h), np.nan)
    direction_12h = np.full(len(high_12h), np.nan)  # 1 for uptrend, -1 for downtrend
    if len(high_12h) >= 10 and not np.isnan(atr_12h).all():
        # Basic upper and lower bands
        hl_avg_12h = (high_12h + low_12h) / 2
        upper_band_12h = hl_avg_12h + 3.0 * atr_12h
        lower_band_12h = hl_avg_12h - 3.0 * atr_12h
        
        # Initialize
        for i in range(len(high_12h)):
            if i == 0:
                supertrend_12h[i] = hl_avg_12h[i]
                direction_12h[i] = 1
            else:
                if close_12h[i] > upper_band_12h[i-1]:
                    direction_12h[i] = 1
                elif close_12h[i] < lower_band_12h[i-1]:
                    direction_12h[i] = -1
                else:
                    direction_12h[i] = direction_12h[i-1]
                
                if direction_12h[i] == 1:
                    supertrend_12h[i] = max(lower_band_12h[i], supertrend_12h[i-1])
                else:
                    supertrend_12h[i] = min(upper_band_12h[i], supertrend_12h[i-1])
    
    # Align Supertrend direction to 4h
    supertrend_dir_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(supertrend_dir_12h_aligned[i]):
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
        volume_filter = volume[i] > vol_ma * 1.5
        
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
            # Look for entries: Donchian breakout + volume + trend filter
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Trend filter: only trade long if 12h Supertrend uptrend, short if downtrend
                trend_filter_long = supertrend_dir_12h_aligned[i] == 1
                trend_filter_short = supertrend_dir_12h_aligned[i] == -1
                
                if bull_breakout and volume_filter and trend_filter_long:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and trend_filter_short:
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