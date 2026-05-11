#!/usr/bin/env python3
name = "6h_PhaseAccumulation_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d trend filter: EMA34 (use close of daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Phase Accumulation indicator (Ehlers) on 6h
    # Uses Hilbert transform via 3-bar momentum for cycle phase
    delta = close - np.roll(close, 1)
    delta[0] = 0
    smooth_delta = pd.Series(delta).ewm(alpha=0.2, adjust=False).mean().values
    
    in_phase = 0.33 * smooth_delta + 0.67 * np.roll(smooth_delta, 1)
    in_phase[0] = 0
    quadrature = smooth_delta - 0.67 * np.roll(in_phase, 1) - 0.33 * np.roll(np.roll(in_phase, 1), 1)
    quadrature[0:2] = 0
    
    # Compute phase (0-2π) and convert to degrees
    # Avoid division by zero
    denom = np.sqrt(in_phase**2 + quadrature**2) + 1e-10
    phase = np.arctan2(quadrature, in_phase)  # -π to π
    phase_degrees = np.degrees(phase)  # -180 to 180
    
    # Normalize to 0-360
    phase_degrees = (phase_degrees + 360) % 360
    
    # Rate of change of phase (angular velocity) - indicates acceleration
    phase_roc = np.roll(phase_degrees, 1) - phase_degrees
    phase_roc[0] = 0
    # Handle wraparound
    phase_roc = np.where(phase_roc > 180, phase_roc - 360, phase_roc)
    phase_roc = np.where(phase_roc < -180, phase_roc + 360, phase_roc)
    
    # Smooth the ROC
    phase_roc_smooth = pd.Series(phase_roc).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Entry conditions:
    # Long: Phase acceleration positive AND in accumulation zone (330-360 or 0-30) AND daily uptrend
    # Short: Phase acceleration negative AND in distribution zone (150-210) AND daily downtrend
    accumulation_zone = (phase_degrees >= 330) | (phase_degrees <= 30)
    distribution_zone = (phase_degrees >= 150) & (phase_degrees <= 210)
    
    long_entry = (phase_roc_smooth > 0) & accumulation_zone & trend_up_1d_aligned
    short_entry = (phase_roc_smooth < 0) & distribution_zone & (~trend_up_1d_aligned)
    
    # Exit conditions: opposite acceleration or trend change
    long_exit = (phase_roc_smooth < 0) | (~trend_up_1d_aligned)
    short_exit = (phase_roc_smooth > 0) | trend_up_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Need enough data for smoothing
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(phase_roc_smooth[i]) or np.isnan(phase_degrees[i]) or
            np.isnan(trend_up_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            if long_entry[i]:
                signals[i] = 0.25
                position = 1
            elif short_entry[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if long_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals