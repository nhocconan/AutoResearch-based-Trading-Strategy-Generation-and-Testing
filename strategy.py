#!/usr/bin/env python3
"""
Experiment #013: 4h Williams %R Extreme + Supertrend + Volume

HYPOTHESIS: Williams %R at extremes (<-80=oversold, >-20=overbought) catches 
reversal points. Supertrend confirms we're not fighting the primary trend. 
Volume spike confirms institutional involvement at reversal points.

WHY THIS SHOULD WORK:
- Bear markets: rallies to Williams %R > -20 are short opportunities
- Bull markets: dips to Williams %R < -80 are long opportunities
- Supertrend keeps us on right side (bullish ATR mode for longs, bearish for shorts)
- Works in ALL conditions because it trades reversals at extremes

TARGET: 75-120 total trades over 4 years (3 symbols ~25-40 each).
DB reference: Williams %R + Supertrend combo proven in volatility strategies.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_williams_r_supertrend_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator for reversals"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    williams = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        window_high = np.max(high[i - period + 1:i + 1])
        window_low = np.min(low[i - period + 1:i + 1])
        range_hl = window_high - window_low
        
        if range_hl > 1e-10:
            williams[i] = -100.0 * (window_high - close[i]) / range_hl
    
    return williams

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend - trend following indicator
    Returns: supertrend value, bullish (1) or bearish (-1)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.zeros(n)
    
    # ATR
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # HL2
    hl2 = (high + low) / 2.0
    
    # Upper and Lower bands
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.full(n, np.nan, dtype=np.float64)
    direction = np.zeros(n, dtype=np.float64)  # 1=bullish, -1=bearish
    
    supertrend[0] = upper_band[0]
    direction[0] = -1
    
    for i in range(1, n):
        prev_close = close[i - 1]
        prev_st = supertrend[i - 1]
        prev_dir = direction[i - 1]
        
        # Bullish trend continues
        if prev_dir == 1:
            if close[i] < lower_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = max(prev_st, lower_band[i])
        else:  # Bearish trend continues
            if close[i] > upper_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = min(prev_st, upper_band[i])
    
    return supertrend, direction

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
    
    # === Calculate 4h indicators ===
    williams_r = calculate_williams_r(high, low, close, period=14)
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_trade = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(williams_r[i]) or np.isnan(supertrend[i]) or np.isnan(atr_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]) or vol_ratio[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        williams = williams_r[i]
        st_dir = st_direction[i]
        vol_spike = vol_ratio[i] > 1.4
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        # LONG: Williams %R < -80 (oversold) + Supertrend bullish + volume
        if not in_position or position_side <= 0:
            # Oversold extreme
            if williams < -80:
                # Supertrend bullish confirmation
                if st_dir > 0:
                    if vol_spike:
                        desired_signal = SIZE
                    elif bars_in_trade >= 4:  # Not too quick to enter
                        desired_signal = SIZE
        
        # SHORT: Williams %R > -20 (overbought) + Supertrend bearish + volume
        if not in_position or position_side >= 0:
            # Overbought extreme
            if williams > -20:
                # Supertrend bearish confirmation
                if st_dir < 0:
                    if vol_spike:
                        desired_signal = -SIZE
                    elif bars_in_trade >= 4:
                        desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT: Williams %R mean reversion ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP when Williams %R mean-reverts (rises above -50)
            if williams > -50:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP when Williams %R mean-reverts (falls below -50)
            if williams < -50:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_in_trade = 0
        else:
            if in_position:
                bars_in_trade += 1
                # Force exit after 20 bars (5 days at 4h) if not stopped out
                if bars_in_trade > 20:
                    in_position = False
                    position_side = 0
                    bars_in_trade = 0
            else:
                bars_in_trade = 0
        
        signals[i] = desired_signal
    
    return signals