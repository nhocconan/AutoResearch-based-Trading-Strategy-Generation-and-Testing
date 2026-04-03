#!/usr/bin/env python3
"""
Experiment #002: 12h Donchian(24) breakout + 1d SMA trend + Volume confirmation + ATR stoploss

HYPOTHESIS: On the 12h timeframe, Donchian breakouts capture significant directional moves.
Filtering by 1d SMA ensures we only trade with the higher timeframe trend, reducing whipsaws.
Volume confirmation adds validity to breakouts. ATR-based trailing stops manage risk.
This combination should produce a moderate number of high-quality trades that perform well in both bull and bear markets by emphasizing trend alignment and proper risk management.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_chop_1d_sma_v3"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    # Calculate True Range (TR) for each bar
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate Exponentially Weighted Moving Average for ATR
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA for trend (Call ONCE before loop) ===
    # Load 1d data ONCE
    try:
        df_1d = get_htf_data(prices, '1d')
    except Exception:
        # Handle potential data loading error if environment is constrained
        return np.zeros(n)

    # Use a simple 30-period SMA for 1d trend confirmation
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=30, min_periods=30).mean().values
    
    # Align HTF data (Shift(1) ensures no look-ahead)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === 12h Indicators ===
    # ATR (for stoploss)
    atr_14 = calculate_atr(high, low, close, period=14)
    # Donchian 24 (for breakout reference)
    dc_upper_24 = pd.Series(high).rolling(window=24, min_periods=24).max().shift(1).values
    dc_lower_24 = pd.Series(low).rolling(window=24, min_periods=24).min().shift(1).values
    
    # Volume MA(20) (for confirmation)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (within 0.20-0.30 range)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50 # Ensure enough data for initial rolling calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if np.isnan(atr_14[i]) or np.isnan(sma_1d_aligned[i]) or np.isnan(dc_upper_24[i]) or np.isnan(dc_lower_24[i]):
            signals[i] = 0.0
            continue
        
        # --- Trend Context ---
        # HTF Trend Check (using aligned data)
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # --- Price Channel Breakout ---
        bullish_breakout = (close[i] > dc_upper_24[i])
        bearish_breakout = (close[i] < dc_lower_24[i])
        
        # --- Volume Confirmation ---
        # Require significant volume spike relative to recent average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # 1. Check Trailing Stop (ATR based)
            if position_side > 0:
                # Stoploss: Price must drop by 2.5 * ATR
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else: # Short position
                # Stoploss: Price must rise by 2.5 * ATR
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # 2. Check Trend Reversal Exit (Only after a minimum hold)
            min_hold = (i - entry_bar) >= 3
            if min_hold:
                if position_side > 0 and htf_bearish:
                    stop_hit = True
                if position_side < 0 and htf_bullish:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                # If still in position, maintain signal
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        
        # Long Entry: Bullish Breakout + Volume Confirm + HTF Bullish Trend
        if bullish_breakout and vol_ok and htf_bullish:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # Short Entry: Bearish Breakout + Volume Confirm + HTF Bearish Trend
        elif bearish_breakout and vol_ok and htf_bearish:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals