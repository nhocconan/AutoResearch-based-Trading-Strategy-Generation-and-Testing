#!/usr/bin/env python3
"""
Experiment #279: 6h Donchian(20) breakout + 12h ATR regime filter + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h aligned with low-volatility regimes (ATR ratio < 0.7) capture high-probability moves. Volume confirmation (>1.5x average) filters weak breakouts. Works in bull markets via breakout continuation and in bear markets via mean reversion at opposite Donchian level. Uses ATR regime to avoid whipsaws in high volatility. Target: 75-150 total trades over 4 years (19-37/year). Uses discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_279_6h_donchian20_12h_atr_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ATR regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ATR(14) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr_12h = np.zeros(len(close_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(close_12h)):
        tr_12h[i] = max(high_12h[i] - low_12h[i], 
                       abs(high_12h[i] - close_12h[i-1]), 
                       abs(low_12h[i] - close_12h[i-1]))
    atr_12h = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate ATR ratio: current ATR / 50-period MA of ATR (volatility regime)
    atr_ma_50 = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.zeros(len(atr_12h))
    atr_ratio[50:] = atr_12h[50:] / atr_ma_50[50:]
    atr_ratio[:50] = 1.0
    
    # Align ATR ratio to 6h timeframe (low ATR ratio = low volatility regime)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    # === 6h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 70  # Enough for 20-period Donchian, 50-period ATR MA, and 14-period ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: Require low volatility (ATR ratio < 0.7) ---
        low_vol_regime = atr_ratio_aligned[i] < 0.7
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on breakout down with volume if volatility increases
                if breakout_down and volume_spike and atr_ratio_aligned[i] > 1.2:
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
                # Exit on breakout up with volume if volatility increases
                if breakout_up and volume_spike and atr_ratio_aligned[i] > 1.2:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require low volatility regime + volume spike + breakout conditions
        if low_vol_regime and volume_spike:
            # Long: breakout up
            if breakout_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout down
            elif breakout_down:
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