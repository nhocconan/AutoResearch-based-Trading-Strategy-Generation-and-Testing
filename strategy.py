#!/usr/bin/env python3
"""
Experiment #021: 4h TRIX Momentum + Volume Spike + ATR Trailing Stop

HYPOTHESIS: TRIX (Triple EMA of rate-of-change) filters market noise by smoothing
through triple smoothing, providing reliable trend signals. Combined with volume
spike confirmation and ATR-based stoploss, this captures sustained momentum moves
while avoiding false breakouts. TRIX crossover is particularly effective because
the triple smoothing reduces lag and false signals compared to single/double EMA.

WHY 4h: Balances trade frequency (20-50/year target) with signal reliability.
Volume spike confirms institutional participation. ATR trailing stop protects
against 2022-style crashes while letting winners run.

KEY CONDITIONS (3 MAX - simplicity = fewer trades = less fee drag):
1. TRIX line crosses signal line (momentum shift)
2. Volume spike (>1.5x 20-bar avg) confirms move
3. ATR-based trailing stop for exit

TARGET: 75-200 total trades over 4 years. HARD MAX: 400.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_vol_spike_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=14, signal=9):
    """
    TRIX: Triple EMA of rate-of-change
    - Triple EMA filters noise
    - TRIX > 0 = bullish momentum
    - TRIX crossing signal line = momentum shift
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Rate of change of triple EMA
    trix = np.zeros(n)
    trix[0] = 0
    for i in range(1, n):
        if ema3.iloc[i-1] != 0:
            trix[i] = ((ema3.iloc[i] - ema3.iloc[i-1]) / ema3.iloc[i-1]) * 100
        else:
            trix[i] = 0
    
    # Smooth TRIX with EMA
    trix_series = pd.Series(trix)
    trix_smooth = trix_series.ewm(span=signal, min_periods=signal, adjust=False).mean().values
    
    return trix, trix_smooth

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.maximum(high[1:] - low[1:], 
                   np.maximum(np.abs(high[1:] - close[:-1]), 
                             np.abs(close[:-1] - low[1:])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.maximum(high[1:] - low[1:], 
                   np.maximum(np.abs(high[1:] - close[:-1]), 
                             np.abs(close[:-1] - low[1:])))
    tr = np.concatenate([[tr[0]], tr])
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === TRIX (local 4h) ===
    trix, trix_signal = calculate_trix(close, period=14, signal=9)
    
    # === ATR for stoploss ===
    atr = calculate_atr(high, low, close, period=14)
    
    # === ADX for trend strength (filter) ===
    adx = calculate_adx(high, low, close, period=14)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Donchian for structure (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    SIZE = 0.30  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # TRIX needs ~50 bars to stabilize
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === INDICATOR VALUES ===
        trix_val = trix[i]
        trix_sig = trix_signal[i]
        trix_prev = trix[i-1] if i > 0 else 0
        trix_sig_prev = trix_signal[i-1] if i > 0 else 0
        
        # TRIX crossover (momentum shift)
        trix_bull_cross = trix_prev < trix_sig_prev and trix_val > trix_sig
        trix_bear_cross = trix_prev > trix_sig_prev and trix_val < trix_sig
        
        # TRIX above/below zero
        trix_above_zero = trix_val > 0
        trix_below_zero = trix_val < 0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ADX TREND FILTER ===
        # ADX > 20 = trending enough for momentum strategy
        adx_trending = adx[i] > 20
        
        # === DONCHIAN BREAKOUT (structure) ===
        price_at_donchian_high = close[i] >= donchian_high[i]
        price_at_donchian_low = close[i] <= donchian_low[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: TRIX crosses above signal + above zero + volume spike + ADX trending
            if trix_bull_cross and trix_above_zero and vol_spike and adx_trending:
                desired_signal = SIZE
            
            # SHORT: TRIX crosses below signal + below zero + volume spike + ADX trending
            if trix_bear_cross and trix_below_zero and vol_spike and adx_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            if position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 6 bars = 1 day) ===
        bars_held = i - entry_bar
        min_hold_bars = 6
        
        if in_position and bars_held >= min_hold_bars:
            # Exit if TRIX reverses
            if position_side > 0 and trix_bear_cross:
                desired_signal = 0.0
            if position_side < 0 and trix_bull_cross:
                desired_signal = 0.0
        
        # === ADX EXIT FILTER (trend exhaustion) ===
        if in_position and adx[i] < 15:
            # Trend weakening - take profits
            if (position_side > 0 and trix_val < trix_sig) or \
               (position_side < 0 and trix_val > trix_sig):
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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