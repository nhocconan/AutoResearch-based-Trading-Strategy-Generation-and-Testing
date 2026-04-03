#!/usr/bin/env python3
"""
Experiment #111: 6h Williams %R + 1d Supertrend + Volume Confirmation

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe.
1d Supertrend(ATR=10, mult=3.0) provides higher-timeframe trend filter to avoid counter-trend trades.
Volume confirmation (1.3x average) ensures institutional participation. This combination
should work in both bull and bear markets by trading mean reversions in the direction
of the 1d trend. Targets 12-37 trades/year on 6h timeframe to minimize fee drag.
Uses discrete position sizing (0.25) and ATR trailing stop (2.5x) for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_1d_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(high)
    if n < atr_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # First TR
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    supertrend[atr_period-1] = lower_band[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, n):
        # Upper Band
        if upper_band[i] < supertrend[i-1] or close[i-1] > supertrend[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = supertrend[i-1]
            
        # Lower Band
        if lower_band[i] > supertrend[i-1] or close[i-1] < supertrend[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = supertrend[i-1]
        
        # Supertrend
        if supertrend[i-1] == upper_band[i-1]:
            if close[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
        else:
            if close[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
    
    return supertrend, direction

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    wr = np.where(denominator != 0, -100 * ((highest_high - close) / denominator), -50.0)
    
    return wr

def generate_signals(prices):
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    close = prices["close"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Supertrend trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    st_1d, st_dir_1d = calculate_supertrend(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        atr_period=10,
        multiplier=3.0
    )
    st_1d_aligned = align_htf_to_ltf(prices, df_1d, st_1d)
    st_dir_1d_aligned = align_htf_to_ltf(prices, df_1d, st_dir_1d)
    
    # === 6h Indicators ===
    williams_r = calculate_williams_r(high, low, close, period=14)
    atr_10 = pd.Series(
        np.maximum(
            np.maximum(high - low, np.abs(high - np.roll(close, 1))),
            np.abs(low - np.roll(close, 1))
        )
    ).ewm(span=10, min_periods=10, adjust=False).mean().values
    atr_10[0] = high[0] - low[0]  # First ATR value
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(atr_10[i]) or 
            np.isnan(st_1d_aligned[i]) or np.isnan(st_dir_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d Supertrend Trend ---
        st_bullish = st_dir_1d_aligned[i] == 1
        st_bearish = st_dir_1d_aligned[i] == -1
        
        # --- Williams %R Conditions ---
        wr_oversold = williams_r[i] <= -80  # Oversold
        wr_overbought = williams_r[i] >= -20  # Overbought
        wr_normal = (williams_r[i] > -80) & (williams_r[i] < -20)  # Normal range
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.3 if vol_ma_20[i] > 1e-10 else False  # 1.3x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_10[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_10[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: WR reversal or opposite extreme
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: WR reaches overbought OR closes below Supertrend
                    if wr_overbought or close[i] < st_1d_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: WR reaches oversold OR closes above Supertrend
                    if wr_oversold or close[i] > st_1d_aligned[i]:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Oversold Williams %R with bullish 1d Supertrend and volume confirmation
        if wr_oversold and st_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Overbought Williams %R with bearish 1d Supertrend and volume confirmation
        elif wr_overbought and st_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals