#!/usr/bin/env python3
"""
Experiment #127: 6h Camarilla pivot levels from 1d + volume confirmation + ADX regime filter

HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) derived from 1d candles,
combined with 6h volume spikes and ADX regime filtering, capture both reversal and continuation moves.
In ranging markets (ADX < 25), fade extremes at R3/S3. In trending markets (ADX >= 25), 
breakout through R4/S4 with volume confirmation. Uses discrete position sizing (0.25) to limit 
drawdown and targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    if len(df_1d) >= 1:
        # Camarilla uses previous day's OHLC
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        
        # Camarilla levels
        R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
        R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
        S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
        S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
        
        # Align to 6h timeframe (shift(1) inside align_htf_to_ltf for completed bars only)
        R4_6h = align_htf_to_ltf(prices, df_1d, R4)
        R3_6h = align_htf_to_ltf(prices, df_1d, R3)
        S3_6h = align_htf_to_ltf(prices, df_1d, S3)
        S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    else:
        R4_6h = R3_6h = S3_6h = S4_6h = np.full(n, np.nan)
    
    # === HTF: 1w data for ADX regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = np.abs(high_1w[1:] - low_1w[1:])
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], tr])
        
        # Directional Movement
        dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                           np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
        dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                            np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        tr_period = 14
        atr = pd.Series(tr).ewm(span=tr_period, min_periods=tr_period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, min_periods=tr_period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, min_periods=tr_period, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=tr_period, min_periods=tr_period, adjust=False).mean().values
        
        adx_6h = align_htf_to_ltf(prices, df_1w, adx)
    else:
        adx_6h = np.full(n, 20.0)  # Default to ranging if insufficient data
    
    # === 6h Indicators ===
    # Volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    valid_ma = ~np.isnan(vol_ma_20)
    vol_ratio[valid_ma] = volume[valid_ma] / vol_ma_20[valid_ma]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(R4_6h[i]) or np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(S4_6h[i]) or 
            np.isnan(adx_6h[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: ADX from 1w ---
        is_trending = adx_6h[i] >= 25
        is_ranging = adx_6h[i] < 25
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions based on regime
            if is_trending:
                # In trending regime: exit when price returns to midpoint (R3/S3)
                if position_side > 0:  # Long
                    if close[i] <= R3_6h[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
                else:  # Short
                    if close[i] >= S3_6h[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
            else:
                # In ranging regime: exit at opposite extreme (S4/R4)
                if position_side > 0:  # Long
                    if close[i] >= R4_6h[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
                else:  # Short
                    if close[i] <= S4_6h[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_ranging:
            # Ranging market: fade extremes at R3/S3
            long_condition = (
                close[i] <= S3_6h[i] and 
                volume_spike
            )
            
            short_condition = (
                close[i] >= R3_6h[i] and 
                volume_spike
            )
        else:
            # Trending market: breakout continuation at R4/S4
            long_condition = (
                close[i] >= R4_6h[i] and 
                volume_spike
            )
            
            short_condition = (
                close[i] <= S4_6h[i] and 
                volume_spike
            )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals