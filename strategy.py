#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d ATR regime filter and volume expansion confirmation.
Uses Camarilla pivot levels from daily timeframe for precise entry/exit levels, combined with
1d ATR-based regime filter to avoid sideways markets. Volume expansion confirms breakout momentum.
Designed for 12h timeframe to capture fewer, higher-quality trades in both bull/bear markets.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Uses discrete position sizing (0.25) to balance return and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate ATR percentage of price for regime filter
    atr_pct_1d = atr_14_1d_aligned / close
    # Regime filter: ATR% > 0.02 (2%) indicates sufficient volatility for breakouts
    volatile_regime = atr_pct_1d > 0.02
    
    # Calculate 1d Camarilla pivot levels (R1, S1, R2, S2)
    # Based on previous 1d bar's OHLC
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels:
    camarilla_r1 = close_1d + (1.0 * range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (1.0 * range_1d * 1.1 / 12)
    camarilla_r2 = close_1d + (1.0 * range_1d * 1.1 / 6)
    camarilla_s2 = close_1d - (1.0 * range_1d * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe (previous 1d bar values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Calculate volume expansion: current volume > 1.8x 30-period MA
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_expansion = volume > 1.8 * vol_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 14)  # need volume MA30 and ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check if we're in a volatile regime (avoid sideways markets)
        if not volatile_regime[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Camarilla R1 AND volume expansion
            if close[i] > camarilla_r1_aligned[i] and volume_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 AND volume expansion
            elif close[i] < camarilla_s1_aligned[i] and volume_expansion[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Camarilla level (S1 for longs, R1 for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S1
                if close[i] < camarilla_s1_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R1
                if close[i] > camarilla_r1_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1dATR_Regime_VolumeExpansion"
timeframe = "12h"
leverage = 1.0