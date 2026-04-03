#!/usr/bin/env python3
"""
Experiment #031: 6h Camarilla Pivot + Volume Spike + Regime Filter (ADX < 25)

HYPOTHESIS: Camarilla pivot levels derived from 1d timeframe provide reliable intraday support/resistance on 6h charts. 
In low volatility regimes (ADX < 25), price tends to respect these levels, offering mean-reversion opportunities at R3/S3 
and breakout continuations at R4/S4. Volume confirmation (>1.5x average) filters false signals. This strategy targets 
range-bound markets which dominate BTC/ETH action in 2025, while avoiding strong trends where pivots fail. 
Discrete position sizing (0.25) and strict entry conditions aim for 12-37 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d bar
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h6 = np.full(n, np.nan)
    camarilla_l6 = np.full(n, np.nan)
    
    if len(df_1d) >= 2:
        # Use previous day's OHLC to avoid look-ahead
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Calculate pivot and ranges
        pivot = (prev_high + prev_low + prev_close) / 3
        range_hl = prev_high - prev_low
        
        # Camarilla levels
        camarilla_h4 = pivot + (range_hl * 1.1 / 2)
        camarilla_l4 = pivot - (range_hl * 1.1 / 2)
        camarilla_h3 = pivot + (range_hl * 1.1 / 4)
        camarilla_l3 = pivot - (range_hl * 1.1 / 4)
        camarilla_h6 = pivot + (range_hl * 1.1 / 6)
        camarilla_l6 = pivot - (range_hl * 1.1 / 6)
        
        # Align to 6h timeframe (shifted by 1 for completed 1d bar)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        camarilla_h6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h6)
        camarilla_l6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l6)
    else:
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_h6_aligned = np.full(n, np.nan)
        camarilla_l6_aligned = np.full(n, np.nan)
    
    # === Regime Filter: ADX(14) on 6h to detect ranging markets (ADX < 25) ===
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_di_14 = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_di_14 = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # DX and ADX
    dx = np.zeros(n)
    dx[:] = np.where((plus_di_14 + minus_di_14) != 0, 
                     np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 
                     0)
    adx_14 = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(adx_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in ranging markets (ADX < 25) ---
        ranging_market = adx_14[i] < 25
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Camarilla Level Conditions ---
        # Mean reversion at H3/L3 (fade extreme moves)
        mean_revert_long = close[i] <= camarilla_l3_aligned[i]
        mean_revert_short = close[i] >= camarilla_h3_aligned[i]
        
        # Breakout continuation at H4/L4 (break strong levels)
        breakout_long = close[i] >= camarilla_h4_aligned[i]
        breakout_short = close[i] <= camarilla_l4_aligned[i]
        
        # --- Exit Logic: Opposite Camarilla level or volume dry-up ---
        if in_position:
            # Exit conditions
            exit_long = (position_side > 0 and 
                        (close[i] >= camarilla_h3_aligned[i] or  # Reached H3 (profit target) or
                         close[i] <= camarilla_l6_aligned[i] or   # Broke L6 (stop) or
                         vol_ratio[i] < 0.8))                     # Volume dried up
            exit_short = (position_side < 0 and 
                         (close[i] <= camarilla_l3_aligned[i] or  # Reached L3 (profit target) or
                          close[i] >= camarilla_h6_aligned[i] or  # Broke H6 (stop) or
                          vol_ratio[i] < 0.8))                    # Volume dried up
            
            if exit_long or exit_short:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Mean reversion at L3 OR breakout at H4 + volume spike + ranging market
        long_condition = (mean_revert_long or breakout_long) and volume_spike and ranging_market
        
        # Short: Mean reversion at H3 OR breakdown at L4 + volume spike + ranging market
        short_condition = (mean_revert_short or breakout_short) and volume_spike and ranging_market
        
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