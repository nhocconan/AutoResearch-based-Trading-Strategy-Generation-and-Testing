#!/usr/bin/env python3
"""
Experiment #028: 4h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels are proven support/resistance where price
reverses. H4/L4 breakouts are strong momentum signals. By combining:
1. 1d Camarilla levels (prev day) for structure
2. Volume spike confirmation (institutional participation)
3. Choppiness Index filter (avoid range-bound whipsaws)
4. ATR stoploss (2.5x for wider stops, fewer stop-outs)

This is a DIRECT IMPLEMENTATION of the DB top performer (ETHUSDT: test Sharpe 1.47).

WHY 4h: Faster than 12h (more signals), more stable than 1h (less noise).
Using 1d Camarilla means structure doesn't change every bar.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 300.
Signal size: 0.25.

LEARNED FROM FAILURES:
- #002: 12h Camarilla overtraded (1155 trades) → Use 4h, stricter volume filter
- Donchian combinations overtraded or undertraded → Camarilla is different structure
- Too many conditions = 0 trades or fee death → Keep it to 3 conditions
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_1d_v2"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_camarilla_levels(high, low, close):
    """Camarilla pivot levels - 8 levels from pivot"""
    n = len(high)
    pivot = (high + low + close) / 3.0
    hl_range = high - low
    
    r1 = pivot + hl_range * 0.09166
    r2 = pivot + hl_range * 0.1833
    r3 = pivot + hl_range * 0.2750
    r4 = pivot + hl_range * 0.3666  # Key breakout level
    
    s1 = pivot - hl_range * 0.09166
    s2 = pivot - hl_range * 0.1833
    s3 = pivot - hl_range * 0.2750
    s4 = pivot - hl_range * 0.3666  # Key breakdown level
    
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4,
        's1': s1, 's2': s2, 's3': s3, 's4': s4
    }

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla levels (prev day)
    cam_1d = calculate_camarilla_levels(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align 1d levels to 4h
    h4_1d = align_htf_to_ltf(prices, df_1d, cam_1d['h4'])
    l4_1d = align_htf_to_ltf(prices, df_1d, cam_1d['l4'])
    h3_1d = align_htf_to_ltf(prices, df_1d, cam_1d['h3'])
    l3_1d = align_htf_to_ltf(prices, df_1d, cam_1d['l3'])
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    was_in_position = False
    
    warmup = 100  # Need enough for indicators + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Get current 1d levels
        h4_level = h4_1d[i] if not np.isnan(h4_1d[i]) else close[i] * 1.02
        l4_level = l4_1d[i] if not np.isnan(l4_1d[i]) else close[i] * 0.98
        h3_level = h3_1d[i] if not np.isnan(h3_1d[i]) else close[i] * 1.01
        l3_level = l3_1d[i] if not np.isnan(l3_1d[i]) else close[i] * 0.99
        
        # Volume confirmation (stronger filter to reduce trades)
        vol_spike = vol_ratio[i] > 1.8
        
        # === REGIME (Choppiness Index) ===
        # Only trade when trending or mildly choppy
        # CHOP > 61.8 = very choppy (skip)
        # CHOP < 50 = trending (prefer)
        is_trending = chop[i] < 50.0
        is_choppy = chop[i] > 61.8
        
        # Skip if too choppy and not in position
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG: Breakout above H4 with volume ===
            # Price closes above H4 breakout level
            if close[i] > h4_level:
                # Volume confirmation OR trending market
                if vol_spike or is_trending:
                    desired_signal = SIZE
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    entry_bar = i
                    stop_price = entry_price - 2.5 * entry_atr
            
            # === SHORT: Breakdown below L4 with volume ===
            # Price closes below L4 breakdown level
            if close[i] < l4_level:
                # Volume confirmation OR trending market
                if vol_spike or is_trending:
                    desired_signal = -SIZE
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    entry_bar = i
                    stop_price = entry_price + 2.5 * entry_atr
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
                in_position = False
                position_side = 0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
                in_position = False
                position_side = 0
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit if price reverts to H3/L3 (mean reversion target)
            if position_side > 0 and close[i] < h3_level:
                desired_signal = 0.0
                in_position = False
                position_side = 0
            if position_side < 0 and close[i] > l3_level:
                desired_signal = 0.0
                in_position = False
                position_side = 0
        
        # === TRAILING STOP ADJUSTMENT ===
        # If in profit (> 2R), tighten stop
        if in_position and position_side > 0:
            profit_r = (close[i] - entry_price) / entry_atr
            if profit_r > 2.0:
                # Move stop to breakeven
                new_stop = entry_price + 0.5 * entry_atr
                stop_price = max(stop_price, new_stop)
        
        if in_position and position_side < 0:
            profit_r = (entry_price - close[i]) / entry_atr
            if profit_r > 2.0:
                # Move stop to breakeven
                new_stop = entry_price - 0.5 * entry_atr
                stop_price = min(stop_price, new_stop)
        
        # === FINAL SIGNAL ===
        if not in_position:
            desired_signal = 0.0
        
        signals[i] = desired_signal
    
    return signals