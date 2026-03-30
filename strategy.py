#!/usr/bin/env python3
"""
Experiment #007: 6h Donchian(16) Breakout + 1d Williams %R Regime + Volume

HYPOTHESIS: Williams %R extremes (<-70 or >-30) on 1d capture market reversals.
When 6h price breaks Donchian(16) WITH volume spike AND Williams %R is at extreme,
the move has high probability of continuation.

WHY IT WORKS IN BULL AND BEAR:
- Williams %R < -70: oversold in any market = bounce candidate
- Williams %R > -30: overbought in any market = dump candidate  
- Donchian breakout confirms momentum, not just oversold/overbought
- Volume spike confirms institutional involvement

TARGET: 75-200 total trades over 4 years = 19-50/year. HARD MAX: 300.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_willr_vol_1d_v2"
timeframe = "6h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R"""
    n = len(close)
    if n < period:
        return np.full(n, -50.0)
    
    willr = np.full(n, -50.0)
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            willr[i] = -50.0
    return willr

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
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Williams %R(14) for regime
    willr_1d = calculate_williams_r(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    willr_1d_aligned = align_htf_to_ltf(prices, df_1d, willr_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (16 periods = ~4 days on 6h)
    donchian_period = 16
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
    
    warmup = 50  # Donchian period
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Williams %R not aligned
        if np.isnan(willr_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Donchian not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME: Williams %R extremes ===
        willr_val = willr_1d_aligned[i]
        oversold = willr_val < -70    # Buy candidates
        overbought = willr_val > -30   # Sell candidates
        
        # === DONCHIAN BREAKOUT ===
        donch_break_high = close[i] > donchian_high[i]  # Breakout above
        donch_break_low = close[i] < donchian_low[i]    # Breakdown below
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + oversold regime + volume ===
            if donch_break_high and oversold and vol_spike:
                desired_signal = SIZE
            # Soft entry: breakout + oversold, no volume (lower conviction)
            elif donch_break_high and oversold:
                desired_signal = SIZE * 0.5
            
            # === SHORT: Breakdown below Donchian low + overbought regime + volume ===
            if donch_break_low and overbought and vol_spike:
                desired_signal = -SIZE
            # Soft entry: breakdown + overbought, no volume
            elif donch_break_low and overbought:
                desired_signal = -SIZE * 0.5
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TIME EXIT: exit after 4 bars (1 day) if no stop hit ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 4:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
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
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals