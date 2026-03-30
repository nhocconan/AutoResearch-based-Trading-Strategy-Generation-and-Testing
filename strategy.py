#!/usr/bin/env python3
"""
Experiment #028: 12h Donchian Breakout + Volume + ATR Stop

HYPOTHESIS: Donchian(20) breakout on 12h captures institutional momentum while
volume confirmation filters false breakouts. ATR-based stops ensure disciplined
risk management. This works in BOTH bull (long breakouts) and bear (short breakdowns).

WHY 12h: Slower than 4h/6h (reduces fee drag), but faster than 1d (more opportunities).
Donchian(20) on 12h = 10-day channel - captures medium-term swings.
Targeting 75-150 total trades over 4 years (19-37/year).

ENTRY LOGIC (3 conditions max):
1. Price breaks above Donchian(20) high OR below Donchian(20) low
2. Volume spike (>1.5x 20-bar MA) confirms the move
3. ATR stoploss at 2.5x (not too tight, not too loose)

KEY INSIGHT: Previous failures had TOO MANY conditions (4-5 filters = too few trades).
This strategy uses only breakout + volume + stoploss = ~75-150 trades expected.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_atr_v1"
timeframe = "12h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = 10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 50  # Need enough for Donchian(20)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        current_high = high[i]
        current_low = low[i]
        
        # Previous bar's Donchian values (avoid look-ahead)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation (at least 1.5x average)
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above previous 20-bar high with volume ===
            if current_high > prev_donchian_high:
                if vol_spike:  # Volume confirmation
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below previous 20-bar low with volume ===
            if current_low < prev_donchian_low:
                if vol_spike:  # Volume confirmation
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
        
        # === PROFIT TARGET CHECK (4R = 10 ATR) ===
        profit_target_hit = False
        
        if in_position and position_side > 0:
            profit_target = entry_price + 4.0 * entry_atr
            if high[i] >= profit_target:
                profit_target_hit = True
        
        if in_position and position_side < 0:
            profit_target = entry_price - 4.0 * entry_atr
            if low[i] <= profit_target:
                profit_target_hit = True
        
        if profit_target_hit:
            # Take profit: reduce position to half
            if position_side > 0:
                desired_signal = SIZE / 2
            else:
                desired_signal = -SIZE / 2
        
        # === MIDDLE CHANNEL EXIT (optional, after min hold) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:  # Hold at least 2 days
            # Exit if price reverts to middle of channel
            if position_side > 0 and close[i] < donchian_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
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