#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_Conservative_v1
Concept: 12h Camarilla pivot R1/S1 breakout with volume confirmation and ATR filter.
- Long: Close breaks above R1 AND volume > 1.5x average AND ATR(12) < ATR(48) (low volatility regime)
- Short: Close breaks below S1 AND volume > 1.5x average AND ATR(12) < ATR(48) (low volatility regime)
- Exit: Close crosses back below R1 (for longs) or above S1 (for shorts)
- Position sizing: 0.25
- Target: 12-37 trades/year (50-150 total over 4 years)
- Works in bull/bear: Pivot levels define support/resistance, volume confirms breakout, ATR filter avoids chop
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Pivot_R1_S1_Breakout_Volume_Conservative_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot levels and volatility context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h: Price data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h: ATR for volatility filter ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12 = pd.Series(tr).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # === 12h: Volume ratio (current vs 24-period average) ===
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma24 > 0, vol_ma24, np.nan)
    
    # === Daily: Pivot levels (using previous day's OHLC) ===
    # Calculate pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 12h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily: ATR for volatility regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_48 = pd.Series(tr_1d).ewm(span=48, adjust=False, min_periods=48).mean().values
    atr_48_aligned = align_htf_to_ltf(prices, df_1d, atr_48)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 48  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_12_val = atr_12[i]
        atr_48_val = atr_48_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_ratio_val) or np.isnan(atr_12_val) or np.isnan(atr_48_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 AND volume confirmation AND low volatility regime
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 1.5
            low_vol = atr_12_val < atr_48_val  # Current volatility below longer-term average
            
            if breakout_long and vol_confirm and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 AND volume confirmation AND low volatility regime
            elif close_val < s1_val and vol_confirm and low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close crosses back below R1
            if close_val < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close crosses back above S1
            if close_val > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals