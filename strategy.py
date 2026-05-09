#!/usr/bin/env python3
# 4h_PhaseShift_Momentum_Reversal
# Hypothesis: Uses Hilbert Transform phase shift to detect momentum exhaustion and reversal points.
# Works in bull/bear markets by identifying overextended moves and mean reversion opportunities.
# Combines phase shift with volume confirmation and volatility filter for high-probability reversals.
# Targets 20-40 trades/year to minimize fee drag while capturing significant moves.

name = "4h_PhaseShift_Momentum_Reversal"
timeframe = "4h"
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
    
    # Hilbert Transform - Phase Calculation (using Ehlers method)
    # Uses 3-bar median filter to reduce noise
    def hilbert_transform(price_series, length=30):
        if len(price_series) < length:
            return np.full_like(price_series, np.nan)
        
        # Median filter for smoothing
        smoothed = np.full_like(price_series, np.nan)
        half_len = length // 2
        for i in range(len(price_series)):
            start_idx = max(0, i - half_len)
            end_idx = min(len(price_series), i + half_len + 1)
            window = price_series[start_idx:end_idx]
            smoothed[i] = np.median(window)
        
        # In-phase and quadrature components
        in_phase = np.full_like(price_series, np.nan)
        quadrature = np.full_like(price_series, np.nan)
        
        # Calculate using delayed signals
        delay = length // 4
        for i in range(delay, len(smoothed)):
            in_phase[i] = smoothed[i - delay]
            quadrature[i] = smoothed[i]
        
        # Calculate phase
        phase = np.full_like(price_series, np.nan)
        valid = (~np.isnan(in_phase)) & (~np.isnan(quadrature)) & (np.abs(in_phase) > 1e-10)
        phase[valid] = np.arctan(quadrature[valid] / in_phase[valid])
        
        # Unwrap phase to avoid jumps
        phase = np.unwrap(phase)
        
        # Calculate rate of phase change (angular velocity)
        angular_velocity = np.full_like(price_series, np.nan)
        for i in range(1, len(phase)):
            if not np.isnan(phase[i]) and not np.isnan(phase[i-1]):
                angular_velocity[i] = phase[i] - phase[i-1]
        
        return angular_velocity
    
    # Calculate phase shift indicator
    phase_shift = hilbert_transform(close, 30)
    
    # Volume confirmation - volume ratio
    def calculate_volume_ratio(vol_series, length=20):
        if len(vol_series) < length:
            return np.full_like(vol_series, np.nan)
        
        vol_ma = np.full_like(vol_series, np.nan)
        # Initialize with simple average
        if len(vol_series) >= length:
            vol_ma[length-1] = np.mean(vol_series[0:length])
            # Exponential smoothing
            for i in range(length, len(vol_series)):
                vol_ma[i] = (vol_ma[i-1] * (length-1) + vol_series[i]) / length
        
        vol_ratio = np.full_like(vol_series, np.nan)
        valid = (~np.isnan(vol_ma)) & (vol_ma > 0)
        vol_ratio[valid] = vol_series[valid] / vol_ma[valid]
        return vol_ratio
    
    volume_ratio = calculate_volume_ratio(volume, 20)
    
    # Volatility filter - ATR ratio
    def calculate_atr(high_series, low_series, close_series, length=14):
        if len(high_series) < length:
            return np.full_like(high_series, np.nan)
        
        tr = np.full_like(high_series, np.nan)
        for i in range(len(high_series)):
            if i == 0:
                tr[i] = high_series[i] - low_series[i]
            else:
                tr[i] = max(
                    high_series[i] - low_series[i],
                    abs(high_series[i] - close_series[i-1]),
                    abs(low_series[i] - close_series[i-1])
                )
        
        atr = np.full_like(high_series, np.nan)
        if len(tr) >= length:
            atr[length-1] = np.mean(tr[0:length])
            for i in range(length, len(tr)):
                atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    atr_ratio = np.full_like(atr, np.nan)
    atr_ma = np.full_like(atr, np.nan)
    
    # Calculate ATR moving average for ratio
    if len(atr) >= 20:
        atr_ma[19] = np.mean(atr[0:20])
        for i in range(20, len(atr)):
            atr_ma[i] = (atr_ma[i-1] * 19 + atr[i]) / 20
    
    valid_atr = (~np.isnan(atr)) & (~np.isnan(atr_ma)) & (atr_ma > 0)
    atr_ratio[valid_atr] = atr[valid_atr] / atr_ma[valid_atr]
    
    # Align indicators to 4h timeframe
    # Phase shift is already calculated on close prices, so no HTF needed
    # But we'll align volume ratio and ATR ratio for consistency
    
    # For volume ratio, we need to calculate it on HTF and align back
    # However, since volume ratio uses the same timeframe, we can use directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(phase_shift[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: negative phase shift (momentum exhaustion) + volume confirmation + volatility expansion
            if (phase_shift[i] < -0.1 and  # Negative angular velocity indicates slowing momentum
                volume_ratio[i] > 1.3 and   # Volume confirmation
                atr_ratio[i] > 0.8):        # Volatility filter (not too low)
                signals[i] = 0.25
                position = 1
            # Enter short: positive phase shift (momentum exhaustion) + volume confirmation + volatility expansion
            elif (phase_shift[i] > 0.1 and   # Positive angular velocity indicates slowing momentum
                  volume_ratio[i] > 1.3 and  # Volume confirmation
                  atr_ratio[i] > 0.8):       # Volatility filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: phase shift turns positive (momentum returning) OR volatility contraction
            if phase_shift[i] > 0.05 or atr_ratio[i] < 0.6:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: phase shift turns negative (momentum returning) OR volatility contraction
            if phase_shift[i] < -0.05 or atr_ratio[i] < 0.6:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals