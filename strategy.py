#!/usr/bin/env python3
"""
4h_PhaseAccumulation_RSI_Trend
4h strategy combining Ehlers Phase Accumulation Cycle indicator with RSI for trend confirmation.
- Long: Phase Accumulation > 0 AND RSI(14) > 50
- Short: Phase Accumulation < 0 AND RSI(14) < 50
- Exit: Opposite signal
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (trend following) and bear markets (mean reversion via RSI extremes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ehlers Phase Accumulation indicator
    def phase_accumulation(close_prices, length=10):
        """Ehlers Phase Accumulation - measures momentum of cycle"""
        if len(close_prices) < length:
            return np.full_like(close_prices, np.nan)
        
        # Smooth the price
        alpha = 1.0 / length
        smoothed = np.zeros_like(close_prices)
        smoothed[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            smoothed[i] = alpha * close_prices[i] + (1 - alpha) * smoothed[i-1]
        
        # Compute differential
        diff = np.diff(smoothed, prepend=smoothed[0])
        
        # Compute in-phase and quadrature components
        alpha1 = math.cos(0.02 * math.pi)  # 36-degree phase shift
        alpha2 = math.sin(0.02 * math.pi)
        
        # Initialize arrays
        in_phase = np.zeros_like(close_prices)
        quadrature = np.zeros_like(close_prices)
        
        # First values
        in_phase[0] = close_prices[0]
        quadrature[0] = 0
        
        # Compute using recursive filter
        for i in range(1, len(close_prices)):
            in_phase[i] = (close_prices[i] - smoothed[i]) * alpha1 + (1 - alpha1) * in_phase[i-1]
            quadrature[i] = diff[i] * alpha2 + (1 - alpha2) * quadrature[i-1]
        
        # Compute phase
        # Avoid division by zero
        denominator = in_phase**2 + quadrature**2
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)
        phase = np.arctan2(quadrature, in_phase)
        
        # Accumulate phase difference
        delta_phase = np.diff(phase, prepend=0)
        # Wrap phase to [-pi, pi]
        delta_phase = np.where(delta_phase > np.pi, delta_phase - 2*np.pi, delta_phase)
        delta_phase = np.where(delta_phase < -np.pi, delta_phase + 2*np.pi, delta_phase)
        
        # Accumulate
        accumulated = np.cumsum(delta_phase)
        
        # Normalize
        return accumulated / length
    
    # Import math for trig functions
    import math
    
    # Calculate Phase Accumulation
    pa = phase_accumulation(close, length=10)
    
    # RSI for trend confirmation
    def rsi(close_prices, length=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        avg_gain[length-1] = np.mean(gain[length-1:2*(length-1)+1]) if 2*(length-1)+1 < len(gain) else np.mean(gain[length-1:])
        avg_loss[length-1] = np.mean(loss[length-1:2*(length-1)+1]) if 2*(length-1)+1 < len(loss) else np.mean(loss[length-1:])
        
        for i in range(length, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, length=14)
    
    # Volume confirmation - 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pa[i]) or np.isnan(rsi_vals[i]) or 
            i < 20 or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend conditions from RSI
        uptrend = rsi_vals[i] > 50
        downtrend = rsi_vals[i] < 50
        
        # Phase Accumulation signals
        pa_bullish = pa[i] > 0
        pa_bearish = pa[i] < 0
        
        if position == 0:
            # Long: bullish phase + uptrend RSI + volume
            if pa_bullish and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: bearish phase + downtrend RSI + volume
            elif pa_bearish and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish phase or RSI turns down
            if pa_bearish or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish phase or RSI turns up
            if pa_bullish or uptrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PhaseAccumulation_RSI_Trend"
timeframe = "4h"
leverage = 1.0