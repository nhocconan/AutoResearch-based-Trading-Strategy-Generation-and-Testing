#!/usr/bin/env python3
"""
Experiment #026: 4h Donchian(20) + Choppiness Regime + Volume Spike

HYPOTHESIS: The Choppiness Index (CHOP) is the optimal regime filter.
It identifies strong directional moves (CHOP < 40) where Donchian breakouts are more reliable.
It avoids ranging markets (CHOP > 61.8) where whipsaws destroy trend strategies.
This strategy seeks trend continuation (Long/Short) only when the market is clearly trending (low CHOP).
It is designed to capture momentum in trending environments while filtering out high-volatility range-bound noise, making it robust in both bull and bear markets by focusing on confirmed breakouts within a defined volatility regime.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_volume_v3"
timeframe = "4h"
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
    # Use .ewm with min_periods to handle initial NaN values gracefully
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP): Measures market range vs volatility.
    CHOP < 38.2 = trending market (Follow breakouts)
    CHOP > 61.8 = ranging market (Avoid)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of true range over the lookback period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            # Calculate TR for each bar in the window
            if j > 0:
                current_tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            else:
                current_tr = high[j] - low[j]
            tr_sum += current_tr
        
        # Highest high - lowest low over the period (Total Range)
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        hl_range = highest_high - lowest_low
        
        if hl_range > 1e-10:
            # CHOP = 100 * log10(sum of TR / HL range) / log10(N)
            chop[i] = 100 * np.log10(tr_sum / hl_range) / np.log10(period)
    
    return chop

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
    # We use the calculated SMA as the HTF trend reference
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === 4h Indicators ===
    # ATR (for stoploss)
    atr_14 = calculate_atr(high, low, close, period=14)
    # Choppiness Index (for regime filter)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian 20 (for breakout reference)
    # Shift(1) ensures we use the previous bar's max/min for the current bar's signal
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20) (for confirmation)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (0.30 is within the 0.20-0.35 range)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50 # Ensure enough data for initial rolling calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if np.isnan(atr_14[i]) or np.isnan(chop_14[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # --- Regime and Trend Context ---
        is_trending = chop_14[i] < 40.0  # Entry only in trending markets
        
        # HTF Trend Check (using aligned data)
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # --- Price Channel Breakout ---
        bullish_breakout = (close[i] > dc_upper_20[i])
        bearish_breakout = (close[i] < dc_lower_20[i])
        
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
        
        # Long Entry: Trending + Bullish Breakout + Volume Confirm
        if is_trending and bullish_breakout and vol_ok and htf_bullish:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # Short Entry: Trending + Bearish Breakout + Volume Confirm
        elif is_trending and bearish_breakout and vol_ok and htf_bearish:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals