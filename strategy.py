#!/usr/bin/env python3
"""
Experiment #011: 6h Camarilla Pivot + Volume Spike + Regime Filter

HYPOTHESIS: Camarilla pivot levels derived from 1d candles provide institutional support/resistance. 
At L3/H3 levels with volume spike (>1.5x mean), we fade the move (mean reversion). 
At L4/H4 levels with volume spike, we breakout in direction of move (continuation). 
Regime filter uses 1d ADX(14) to avoid whipsaw: only trade when ADX > 20 (trending) OR ADX < 20 (ranging) 
with appropriate logic. This adaptive approach works in both bull and bear markets by aligning 
with institutional order flow at key levels. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_011_6h_camarilla_pivot_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels and ADX ===
    df_1d = get_htf_data(prices, '1d')
    # Typical Price for 1d
    tp_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    # Camarilla calculation uses previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First value invalid
    
    # Camarilla levels
    camarilla_h5 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) * 1.5 / 4
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) * 1.25 / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) * 1.25 / 4
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) * 1.5 / 4
    camarilla_l5 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align HTF data to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for 1d indicator stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: ADX > 20 = trending, ADX < 20 = ranging ---
        adx_val = adx_1d_aligned[i]
        is_trending = adx_val > 20
        is_ranging = adx_val < 20
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Camarilla Logic ---
        # Fade at H3/L3 (mean reversion in ranging markets)
        fade_long = is_ranging and volume_spike and (price <= camarilla_l3_aligned[i])
        fade_short = is_ranging and volume_spike and (price >= camarilla_h3_aligned[i])
        
        # Breakout continuation at H4/L4 (trend continuation in trending markets)
        breakout_long = is_trending and volume_spike and (price >= camarilla_h4_aligned[i])
        breakout_short = is_trending and volume_spike and (price <= camarilla_l4_aligned[i])
        
        # --- Exit Logic: Opposite signal or volume drought ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_long = (position_side > 0) and (
                fade_short or breakout_short or  # Opposite signal
                (bars_since_entry >= 4)          # Time-based exit
            )
            exit_short = (position_side < 0) and (
                fade_long or breakout_long or    # Opposite signal
                (bars_since_entry >= 4)          # Time-based exit
            )
            
            if exit_long or exit_short:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Fade at L3 OR Breakout at H4
        if fade_long or breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Fade at H3 OR Breakout at L4
        elif fade_short or breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals