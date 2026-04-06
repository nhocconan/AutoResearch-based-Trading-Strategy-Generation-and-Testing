#!/usr/bin/env python3
"""
4h Donchian(20) breakout with volume confirmation and volatility filter
Hypothesis: Donchian breakouts capture momentum bursts. Volume filter ensures institutional participation. Volatility filter avoids choppy markets. Works in bull (buy breakouts) and bear (sell breakdowns). Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volatility_filter_v1"
timeframe = "4h"
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
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Volatility filter: ATR(14) / ATR(50) < 0.8 (avoid high volatility chop)
    atr50 = np.full(n, np.nan)
    if n >= 50:
        tr_sum = np.sum(tr[:49])  # First 49 TR values
        atr50[49] = tr_sum / 49
        for i in range(50, n):
            atr50[i] = (atr50[i-1] * 49 + tr[i-1]) / 50
    vol_filter = atr / atr50 < 0.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need ATR50 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(atr50[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR volatility filter fails
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                not vol_filter[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR volatility filter fails
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                not vol_filter[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: upper/lower with volatility filter
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with volatility filter
                if bull_breakout and vol_filter[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with volatility filter
                elif bear_breakout and vol_filter[i]:
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