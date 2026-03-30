#!/usr/bin/env python3
"""
Experiment #006 v2: 4h TRIX Momentum + Donchian Breakout + Volume Confirm

HYPOTHESIS: Combine TRIX (proven momentum in ETHUSDT test Sharpe 1.32) with 
Donchian breakout (proven in multiple DB winners) for a tighter, more robust signal.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: TRIX positive crossover + Donchian up + volume = strong long entries
- Bear: TRIX negative crossover + Donchian down + volume = strong short entries
- Choppiness filter prevents range-bound whipsaws
- ATR stoploss manages risk symmetrically in both directions

KEY DESIGN (learned from failures):
- TRIX(15) instead of HMA for smoother momentum signal
- Donchian(20) for clear price structure
- Volume 2.0x threshold (tighter than 1.8x to reduce trades)
- Choppiness < 50 (same as current best)
- 2.5 ATR stoploss
- Min hold 4 bars to reduce fee churn

TARGET: 100-180 total trades over 4 years (25-45/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_donchian_volume_v2"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=15):
    """
    TRIX - Triple EMA oscillator
    1 = EMA1, 2 = EMA2, 3 = EMA3 over close
    TRIX = 100 * (EMA3_current - EMA3_prev) / EMA3_prev
    Positive TRIX = upward momentum, Negative = downward
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    # Triple EMA
    ema1 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # TRIX rate of change
    trix = np.zeros(n, dtype=np.float64)
    ema3_vals = ema3.values
    
    for i in range(1, n):
        if not np.isnan(ema3_vals[i]) and not np.isnan(ema3_vals[i-1]) and abs(ema3_vals[i-1]) > 1e-10:
            trix[i] = 100 * (ema3_vals[i] - ema3_vals[i-1]) / ema3_vals[i-1]
        else:
            trix[i] = 0.0
    
    return trix

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
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 50 = trending - GOOD to enter
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Local 4h indicators ===
    trix = calculate_trix(close, period=15)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_series = pd.Series(trix)
    trix_signal = trix_series.ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Volume ratio (20-period MA) - 2.0x threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 250  # 200 for donchian + 45 for TRIX triple EMA + 20 for vol MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_trending = chop_value < 50
        
        # === VOLUME CONFIRMATION (2.0x) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        
        breakout_up = close[i] > prev_donchian_up
        breakout_down = close[i] < prev_donchian_lo
        
        # === TRIX MOMENTUM SIGNAL ===
        # Long: TRIX crosses above signal line (positive momentum building)
        # Short: TRIX crosses below signal line (negative momentum building)
        prev_trix = trix[i - 1]
        prev_signal = trix_signal[i - 1]
        curr_trix = trix[i]
        curr_signal = trix_signal[i]
        
        trix_cross_up = (prev_trix < prev_signal) and (curr_trix > curr_signal)
        trix_cross_down = (prev_trix > prev_signal) and (curr_trix < curr_signal)
        
        # TRIX momentum direction
        trix_positive = curr_trix > 0
        trix_negative = curr_trix < 0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX bullish cross + Donchian up + volume + trending ===
            # Require: TRIX positive OR bullish cross, breakout up, volume, trending
            long_condition = (triy_cross_up or trix_positive) and breakout_up and vol_spike and is_trending
            if long_condition:
                desired_signal = SIZE
            
            # === SHORT: TRIX bearish cross + Donchian down + volume + trending ===
            short_condition = (trix_cross_down or trix_negative) and breakout_down and vol_spike and is_trending
            if short_condition:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: exit if price falls 2.5 ATR from recent high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if TRIX turns negative
                if trix_negative and not trix_positive:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: exit if price rises 2.5 ATR from recent low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if TRIX turns positive
                if trix_positive and not trix_negative:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals