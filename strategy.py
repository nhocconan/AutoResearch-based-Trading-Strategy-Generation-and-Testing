#!/usr/bin/env python3
"""
Experiment #035: 6h Donchian Breakout + Weekly Pivot Direction + Volume Filter

HYPOTHESIS: Uses 6h Donchian channel breakouts (20-period) for entry timing,
filtered by weekly pivot direction (bullish/bearish based on price vs weekly pivot)
and volume confirmation. In strong weekly trends (price above/below weekly pivot),
we take 6h Donchian breakouts. Uses ATR-based stoploss (2.0) and discrete
position sizing (0.25) to minimize fee drag. Target: 75-150 trades over 4 years
on 6h timeframe with proper risk control for both bull and bear markets.
Weekly pivot provides structural bias that works across market regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_035_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: Weekly data for pivot calculation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # We'll use the basic pivot (P) for direction filter
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Align to 6h timeframe (with shift(1) for completed weekly bars only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === 6h Indicators: ATR(14) for stoploss and volume filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume ratio: current volume / 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Sufficient warmup for Donchian and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Weekly Pivot Direction Filter ---
        # Bullish bias: price above weekly pivot
        # Bearish bias: price below weekly pivot
        is_bullish_bias = price > weekly_pivot_aligned[i]
        is_bearish_bias = price < weekly_pivot_aligned[i]
        
        # --- Volume Confirmation ---
        # Require volume > 1.5x average for breakout validity
        vol_confirmed = vol_ratio[i] > 1.5
        
        # --- Breakout Detection ---
        # Bullish breakout: price closes above Donchian high
        # Bearish breakout: price closes below Donchian low
        bullish_breakout = close[i] > donchian_high[i-1]  # Close above prior high
        bearish_breakout = close[i] < donchian_low[i-1]   # Close below prior low
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if bias reverses or volume dries up
                if not is_bullish_bias or not vol_confirmed:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if bias reverses or volume dries up
                if not is_bearish_bias or not vol_confirmed:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Take profit at Donchian midpoint on bias reversal
            if position_side > 0 and price < donchian_mid[i] and not is_bullish_bias:
                signals[i] = 0.0  # Exit long at midpoint
                in_position = False
                position_side = 0
                bars_since_entry = 0
                continue
            elif position_side < 0 and price > donchian_mid[i] and not is_bearish_bias:
                signals[i] = 0.0  # Exit short at midpoint
                in_position = False
                position_side = 0
                bars_since_entry = 0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Bullish breakout + bullish bias + volume confirmation
        if bullish_breakout and is_bullish_bias and vol_confirmed:
            # Enter on retest of breakout level (donchian_high)
            if abs(price - donchian_high[i-1]) < 0.5 * atr_14[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
        # Short: Bearish breakout + bearish bias + volume confirmation
        elif bearish_breakout and is_bearish_bias and vol_confirmed:
            # Enter on retest of breakout level (donchian_low)
            if abs(price - donchian_low[i-1]) < 0.5 * atr_14[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
    
    return signals

def calculate_hma(arr, period):
    """Hull Moving Average - kept for compatibility but not used"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=np.float64)
    
    # Calculate WMA for period/2
    half_period = max(1, period // 2)
    wma_half = np.zeros_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start_idx = max(0, i - half_period + 1)
        weights = np.arange(1, i - start_idx + 2)
        wma_half[i] = np.dot(arr[start_idx:i+1], weights) / weights.sum()
    
    # Calculate WMA for full period
    wma_full = np.zeros_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start_idx = max(0, i - period + 1)
        weights = np.arange(1, i - start_idx + 2)
        wma_full[i] = np.dot(arr[start_idx:i+1], weights) / weights.sum()
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Calculate final WMA of raw HMA with sqrt(period)
    sqrt_period = max(1, int(np.sqrt(period)))
    hma = np.zeros_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start_idx = max(0, i - sqrt_period + 1)
        weights = np.arange(1, i - start_idx + 2)
        hma[i] = np.dot(raw_hma[start_idx:i+1], weights) / weights.sum()
    
    return hma