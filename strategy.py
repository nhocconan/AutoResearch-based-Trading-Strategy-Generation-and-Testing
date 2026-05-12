#!/usr/bin/env python3
# 6H_PivotBreakout_VolumeRegime
# Hypothesis: Combines daily Camarilla pivot levels with volume confirmation and volatility regime filter.
# Long when price breaks above R4 with volume spike and volatility expansion; short when breaks below S4.
# Uses volatility regime (ATR ratio) to avoid whipsaws in low volatility environments.
# Targets 15-25 trades/year to minimize fee drag while capturing strong breakouts.

name = "6H_PivotBreakout_VolumeRegime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility regime filter (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values  # Longer MA for regime
    
    # Volume spike detector (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's high, low, close
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Previous day values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(d_high, 1)
    prev_low = np.roll(d_low, 1)
    prev_close = np.roll(d_close, 1)
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else 0
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else 0
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else 0
    
    # Calculate Camarilla levels
    # R4 = Close + ((High - Low) * 1.1/2)
    # S4 = Close - ((High - Low) * 1.1/2)
    rng = prev_high - prev_low
    camarilla_r4 = prev_close + (rng * 1.1 / 2)
    camarilla_s4 = prev_close - (rng * 1.1 / 2)
    
    # Align to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(atr[i]) or np.isnan(atr_ma[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: volatility expansion (current ATR > 1.1x MA)
        vol_expansion = atr[i] > atr_ma[i] * 1.1
        
        # Volume confirmation: current volume > 1.5x MA
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # LONG: Break above R4 with volume spike and volatility expansion
            if close[i] > camarilla_r4_aligned[i] and vol_spike and vol_expansion:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S4 with volume spike and volatility expansion
            elif close[i] < camarilla_s4_aligned[i] and vol_spike and vol_expansion:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to midpoint or volatility collapses
            midpoint = (camarilla_r4_aligned[i] + camarilla_s4_aligned[i]) / 2
            vol_contraction = atr[i] < atr_ma[i] * 0.9
            if close[i] < midpoint or vol_contraction:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to midpoint or volatility collapses
            midpoint = (camarilla_r4_aligned[i] + camarilla_s4_aligned[i]) / 2
            vol_contraction = atr[i] < atr_ma[i] * 0.9
            if close[i] > midpoint or vol_contraction:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals