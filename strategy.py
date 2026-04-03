#!/usr/bin/env python3
"""
Experiment #039: 6h Williams %R + 12h ADX Trend + Volume Confirmation
HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h, while 12h ADX(14) filters for trending markets. Volume confirmation ensures breakout validity. Designed to capture mean reversals in ranging markets (ADX < 25) and trend continuations in strong trends (ADX >= 25). Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_039_6h_williamsr_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ADX(14) on 12h
    def calculate_adx(high, low, close, period=14):
        if len(high) < period:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        # Smoothed TR, DM+
        tr_period = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False).mean().values
        dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False).mean().values
        # Directional Indicators
        di_plus = 100 * dm_plus_period / tr_period
        di_minus = 100 * dm_minus_period / tr_period
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        # Handle division by zero
        adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
        return adx
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Williams %R(14) ===
    def calculate_williams_r(high, low, close, period=14):
        if len(high) < period:
            return np.full_like(high, np.nan)
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
        return williams_r
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Warmup for indicator stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(adx_12h_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Williams %R Conditions ---
        wr_oversold = williams_r[i] < -80  # Oversold
        wr_overbought = williams_r[i] > -20  # Overbought
        
        # --- ADX Trend Regime ---
        adx_value = adx_12h_aligned[i]
        is_trending = adx_value >= 25  # Strong trend
        is_ranging = adx_value < 25  # Ranging/weak trend
        
        # --- Volume Confirmation ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 2R
                if high[i] >= entry_price + 2 * 2.5 * atr_14[i]:
                    signals[i] = position_side * SIZE * 0.5  # Half position
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 2R
                if low[i] <= entry_price - 2 * 2.5 * atr_14[i]:
                    signals[i] = position_side * SIZE * 0.5  # Half position
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Exit conditions based on regime
            if is_ranging:
                # In ranging market: mean reversion - exit when Williams %R returns to neutral
                if position_side > 0 and williams_r[i] >= -50:  # Long exit
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and williams_r[i] <= -50:  # Short exit
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:
                # In trending market: trend continuation - exit when Williams %R reaches extreme
                if position_side > 0 and williams_r[i] > -20:  # Long exit (overbought)
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and williams_r[i] < -80:  # Short exit (oversold)
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_ranging:
            # Ranging market: mean reversion trades
            if wr_oversold and volume_spike:
                # Long from oversold
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif wr_overbought and volume_spike:
                # Short from overbought
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Trending market: trend continuation trades
            if wr_oversold and volume_spike:
                # Long from oversold in uptrend (pullback)
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif wr_overbought and volume_spike:
                # Short from overbought in downtrend (pullback)
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals