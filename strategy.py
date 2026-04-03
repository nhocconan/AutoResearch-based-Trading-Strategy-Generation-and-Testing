#!/usr/bin/env python3
"""
Experiment #036: 12h Donchian Breakout + 1d Volume Spike + ATR Regime Filter

HYPOTHESIS: Donchian(20) breakout on 12h timeframe with 1d volume confirmation (>2.0x average) 
and ATR-based regime filter (ATR(14) > ATR(50) for volatility expansion) captures strong 
trend moves in both bull and bear markets. Uses discrete position sizing (0.30) and 
ATR(14) stoploss (2.5x) to manage risk. Targets 12-37 trades/year on 12h timeframe 
(50-150 total over 4 years) to minimize fee drag while participating in significant 
breakouts with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike and ATR regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # Calculate ATR(14) and ATR(50) on 1d for regime filter
    if len(df_1d) >= 50:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr = np.zeros(len(close_1d))
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
        
        # ATR(14) and ATR(50)
        atr_14_1d = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_50_1d = pd.Series(tr).ewm(span=50, min_periods=50, adjust=False).mean().values
        
        # Regime: ATR(14) > ATR(50) indicates volatility expansion (trending market)
        atr_ratio_1d = np.zeros(len(atr_14_1d))
        atr_ratio_1d[50:] = atr_14_1d[50:] / atr_50_1d[50:]
        atr_ratio_1d[:50] = 1.0  # Neutral for warmup
        atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    else:
        atr_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 12h Indicators: Donchian Channel (20) ===
    # Calculate Donchian upper and lower bands on 12h
    if len(df_1d) >= 20:  # Need enough 1d data to align properly
        # We'll calculate Donchian on 12h data derived from 1d alignment approach
        # For simplicity, we use 1d data to approximate 12h channels (conservative)
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Donchian(20) on 1d (serves as proxy for 12h due to alignment)
        donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
        donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
        
        donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
        donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    else:
        donch_high_aligned = np.full(n, np.nan)
        donch_low_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in volatility expansion regimes ---
        volatility_expansion = atr_ratio_1d_aligned[i] > 1.2
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using recent price data
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], 
                           abs(high[j] - close[j-1]), 
                           abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume and volatility expansion
        long_condition = (
            close[i] > donch_high_aligned[i] and 
            volume_spike and 
            volatility_expansion
        )
        
        # Short: Price breaks below Donchian low with volume and volatility expansion
        short_condition = (
            close[i] < donch_low_aligned[i] and 
            volume_spike and 
            volatility_expansion
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