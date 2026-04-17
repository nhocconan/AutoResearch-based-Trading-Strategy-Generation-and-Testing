#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_MeanReversion_v1
Camarilla pivot levels from 1d: long at S1 with volume spike in mean-reversion regime,
short at R1 with volume spike in mean-reversion regime.
Exit at opposite pivot level or middle (close).
Uses 1d Bollinger Band width percentile < 50 to identify mean-reversion regime.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === 1d data for Camarilla pivot and regime ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # Using previous day's H, L, C to avoid lookahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    R4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    P = (prev_high + prev_low + prev_close) / 3  # pivot point
    
    # Align to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    P_4h = align_htf_to_ltf(prices, df_1d, P)
    
    # === Mean-reversion regime filter: Bollinger Band width < 50th percentile ===
    # Use 20-period BB on 1d closes
    bb_middle = pd.Series(prev_close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(prev_close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Percentile of BB width (252-day lookback for stability)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align regime filter to 4h
    bb_width_percentile_4h = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    mean_reversion_regime = bb_width_percentile_4h < 50  # True when in mean-reversion regime
    
    # === Volume spike detector (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)  # Volume at least 2x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(P_4h[i]) or np.isnan(mean_reversion_regime[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long at S1 with volume spike in mean-reversion regime
            if (close[i] <= S1_4h[i] and 
                volume_spike[i] and 
                mean_reversion_regime[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short at R1 with volume spike in mean-reversion regime
            elif (close[i] >= R1_4h[i] and 
                  volume_spike[i] and 
                  mean_reversion_regime[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price reaches P (pivot point) or R1
            if close[i] >= P_4h[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches P (pivot point) or S1
            if close[i] <= P_4h[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0