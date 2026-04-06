#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + Volume Spike + DMI Trend Filter + ATR Stoploss
Hypothesis: Donchian breakouts on 12h timeframe with volume confirmation (>2x average) and strong trend (DMI+ > DMI-) capture trends while avoiding whipsaws. DMI is less prone to whipsaw than ADX in choppy markets. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_volume_dmi_v1"
timeframe = "12h"
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
    
    # 14-period DMI (+DI, -DI)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    if n >= 14:
        # +DM and -DM
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True Range (same as ATR calculation)
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        
        # Smoothed values
        tr14 = np.full(n, np.nan)
        plus_dm14 = np.full(n, np.nan)
        minus_dm14 = np.full(n, np.nan)
        
        if len(tr) >= 14:
            tr14[14] = np.sum(tr[:14])
            plus_dm14[14] = np.sum(plus_dm[:14])
            minus_dm14[14] = np.sum(minus_dm[:14])
            
            for i in range(15, n):
                tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i-1]
                plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i-1] / 14) + plus_dm[i-1]
                minus_dm14[i] = minus_dm14[i-1] - (minus_dm14[i-1] / 14) + minus_dm[i-1]
        
        # Directional Indicators
        valid = ~np.isnan(tr14) & (tr14 != 0)
        if np.any(valid):
            plus_di[valid] = 100 * plus_dm14[valid] / tr14[valid]
            minus_di[valid] = 100 * minus_dm14[valid] / tr14[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian and DMI
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]):
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
        
        # DMI trend filter (+DI > -DI indicates bullish momentum)
        trend_filter = plus_di[i] > minus_di[i]
        
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
            # Look for entries: Donchian breakout + volume + DMI trend filter
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                if bull_breakout and volume_filter and trend_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and not trend_filter:  # For short, we want -DI > +DI
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