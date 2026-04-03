#!/usr/bin/env python3
"""
Experiment #1507: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 6h capture medium-term swings, filtered by 1d weekly pivot levels (R1/S1) for direction and volume confirmation (>1.5x average) to reduce false breakouts. Weekly pivot provides structural support/resistance that works in both bull and bear markets. ATR-based stoploss (2.0) manages risk. Target: 12-37 trades/year (50-150 total over 4 years) by using tight entry conditions and multi-timeframe confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1507_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from prior week (using last 5 trading days approximation)
    # For simplicity, use prior day's high/low/close to calculate classic pivot points
    # In practice, weekly pivot would use prior week's OHLC, but we approximate with daily
    # Shift by 1 to use prior day's data (no look-ahead)
    if len(high_1d) >= 2:
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close = np.roll(close_1d, 1)
        prev_high[0] = high_1d[0]  # first bar uses current day's high
        prev_low[0] = low_1d[0]
        prev_close[0] = close_1d[0]
    else:
        prev_high = high_1d
        prev_low = low_1d
        prev_close = close_1d
    
    # Classic pivot points: P = (H + L + C)/3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Determine bias: price above pivot = bullish bias, below = bearish bias
    bias = np.where(close_1d > pivot, 1, np.where(close_1d < pivot, -1, 0))
    bias_aligned = align_htf_to_ltf(prices, df_1d, bias)
    
    # === 6h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(bias_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d bias alignment from pivot
        bias_direction = bias_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if bias_direction != 0 and volume_spike:
            # Breakout: price breaks above upper band OR below lower band
            if price > donch_high[i] and bias_direction > 0:  # Bullish bias breakout
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and bias_direction < 0:  # Bearish bias breakdown
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals