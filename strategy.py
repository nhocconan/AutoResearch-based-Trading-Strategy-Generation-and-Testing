#!/usr/bin/env python3
"""
Experiment #385: 12h Donchian Breakout + 1d Volume Spike + ATR Regime Filter

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, confirmed by 1d volume spike (>2x average) 
and filtered by ATR-based volatility regime (ATR(14) > ATR(50) for trending markets), 
captures strong momentum moves while avoiding choppy periods. Uses discrete position sizing 
(0.25) and ATR(14) stoploss (2.5x) to manage risk. Targets 12-37 trades/year on 12h 
timeframe (50-150 total over 4 years) to minimize fee drag while allowing for trend 
following in both bull and bear markets. The 1d volume filter ensures institutional 
participation, and the ATR regime filter avoids false breakouts in low volatility environments.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
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
    
    # Calculate 1d volume ratio (current vs 20-period average)
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    if len(df_1d) >= 50:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], 
                           abs(high_1d[i] - close_1d[i-1]), 
                           abs(low_1d[i] - close_1d[i-1]))
        
        atr_14_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_50_1d = pd.Series(tr_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        
        # Regime: trending when ATR(14) > ATR(50) (increasing volatility)
        atr_regime_1d = atr_14_1d > atr_50_1d
        atr_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_1d.astype(np.float64))
    else:
        atr_regime_1d_aligned = np.full(n, 1.0)  # Default to trending if insufficient data
    
    # === 12h Indicators ===
    # Calculate Donchian channels (20-period) on 12h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:  # Need 20 periods including current
            start_idx = max(0, i - 19)
            highest_20[i] = np.max(high[start_idx:i+1])
            lowest_20[i] = np.min(low[start_idx:i+1])
        else:
            highest_20[i] = np.nan
            lowest_20[i] = np.nan
    
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
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(atr_regime_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Regime Filter: Only trade in increasing volatility regimes ---
        vol_regime = atr_regime_1d_aligned[i] > 0.5  # Boolean as float
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for 12h timeframe
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
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
        # Long: Price breaks above Donchian upper band with volume and regime confirmation
        long_condition = (
            close[i] > highest_20[i] and 
            volume_spike and 
            vol_regime
        )
        
        # Short: Price breaks below Donchian lower band with volume and regime confirmation
        short_condition = (
            close[i] < lowest_20[i] and 
            volume_spike and 
            vol_regime
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