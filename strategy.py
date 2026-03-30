#!/usr/bin/env python3
"""
Experiment #028: 4h Donchian Breakout + Volume Spike + 12h EMA Trend

HYPOTHESIS: Tight Donchian breakout with dual confirmation (volume + HTF trend)
will capture institutional moves while avoiding whipsaws. 4h is proven timeframe.
Combining BOTH filters (AND logic) prevents overtrading from loose conditions.

WHY IT WORKS: Donchian(20) = 5-day channel - captures medium-term institutional
breakouts. Volume spike confirms institutional participation. 12h EMA(50) ensures
we trade WITH the higher timeframe trend. CHOP < 50 keeps us out of range-bound
markets. Only long when price above HTF EMA, only short when below.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Previous strategies overtraded due to OR logic between filters.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_ema50_12h_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(50) for trend direction
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (20 periods = 3.3 days on 4h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume metrics
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
    
    warmup = 100  # Need enough for Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (12h EMA50) ===
        price_above_12h_ema = close[i] > ema_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_12h_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # CHOP < 50 = trending (good for entries)
        # CHOP > 61.8 = choppy (skip)
        is_trending = chop[i] < 50.0
        is_choppy = chop[i] > 61.8
        
        # Skip new entries if too choppy
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # Previous bar values
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # Volume spike confirmation (need STRONG volume = 2.0x)
        vol_spike = vol_ratio[i] > 2.0
        
        # Close above/below previous Donchian high/low (strict confirmation)
        close_breaks_high = close[i] > prev_donchian_high
        close_breaks_low = close[i] < prev_donchian_low
        
        # === ENTRY LOGIC (TIGHT - ALL conditions must pass) ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Close breaks above Donchian high + BOTH filters ===
            # Must have: close above Donchian + volume spike + above HTF EMA + trending regime
            if close_breaks_high and price_above_12h_ema:
                if vol_spike and is_trending:
                    desired_signal = SIZE
            
            # === SHORT: Close breaks below Donchian low + BOTH filters ===
            # Must have: close below Donchian + volume spike + below HTF EMA + trending regime
            if close_breaks_low and price_below_12h_ema:
                if vol_spike and is_trending:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 8 bars = 1.3 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit on opposite Donchian touch
            if position_side > 0 and close[i] < prev_donchian_low:
                desired_signal = 0.0
            if position_side < 0 and close[i] > prev_donchian_high:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals