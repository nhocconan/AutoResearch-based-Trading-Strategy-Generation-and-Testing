#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h Supertrend + Volume Confirmation
Hypothesis: Donchian breakout captures momentum, 12h Supertrend filters trend direction, volume confirms breakout strength.
Long when price breaks above Donchian upper band (20-period) AND price > 12h Supertrend AND volume > 1.5x average.
Short when price breaks below Donchian lower band AND price < 12h Supertrend AND volume > 1.5x average.
ATR-based stoploss and position sizing 0.25 to limit drawdown.
Works in bull (breakouts with trend) and bear (breakdowns with trend). Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14383_4h_donchian20_12h_supertrend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Supertrend (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Supertrend parameters (12h)
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for 12h
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate upper and lower bands
    hl2_12h = (high_12h + low_12h) / 2
    upper_band = hl2_12h + (multiplier * atr_12h)
    lower_band = hl2_12h - (multiplier * atr_12h)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr_12h[i]) or atr_12h[i] == 0:
            supertrend[i] = supertrend[i-1] if i > 0 else hl2_12h[i]
            direction[i] = direction[i-1] if i > 0 else 1
            continue
            
        if close_12h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_12h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require at least 150% of average volume
    
    # ATR for stoploss (4h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_window, atr_period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR Supertrend turns bearish OR stoploss
            if (close[i] <= donchian_low[i] or 
                close[i] <= supertrend_aligned[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR Supertrend turns bullish OR stoploss
            if (close[i] >= donchian_high[i] or 
                close[i] >= supertrend_aligned[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + Supertrend direction + volume
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            supertrend_bullish = close[i] > supertrend_aligned[i]
            supertrend_bearish = close[i] < supertrend_aligned[i]
            
            long_setup = long_breakout and supertrend_bullish and vol_filter[i]
            short_setup = short_breakout and supertrend_bearish and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals