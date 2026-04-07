#!/usr/bin/env python3
"""
6h_adx_elliott_wave_v1
Hypothesis: Combines ADX trend strength with Elliott Wave pattern recognition (5-wave impulse + 3-wave correction).
In trending markets (ADX>25), enter on wave 3 or wave 5 impulses; in ranging markets (ADX<20), fade wave A/C corrections.
Works in bull markets by catching impulse waves up, in bear markets by catching impulse waves down.
Target: 15-30 trades/year by requiring ADX>25 for trend trades and clear wave structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_elliott_wave_v1"
timeframe = "6h"
leverage = 1.0

def _wma(arr, period):
    """Weighted Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    weights = np.arange(1, period + 1, dtype=float)
    weights_sum = weights.sum()
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(period - 1, len(arr)):
        result[i] = np.dot(arr[i - period + 1:i + 1], weights) / weights_sum
    return result

def _calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = _wma(tr, period)
    dm_plus_period = _wma(dm_plus, period)
    dm_minus_period = _wma(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = _wma(dx, period)
    
    return adx

def _elliott_wave_signal(close, lookback=50):
    """
    Detect Elliott Wave patterns:
    Returns: 1 for bullish impulse (wave 3/5), -1 for bearish impulse (wave 3/5), 0 otherwise
    Simplified: looks for 3-5 wave momentum structure
    """
    if len(close) < lookback:
        return 0
    
    # Look for momentum waves in recent price action
    window = close[-lookback:]
    n = len(window)
    
    # Find local peaks and troughs
    peaks = []
    troughs = []
    
    for i in range(2, n-2):
        if window[i] > window[i-1] and window[i] > window[i-2] and \
           window[i] > window[i+1] and window[i] > window[i+2]:
            peaks.append(i)
        if window[i] < window[i-1] and window[i] < window[i-2] and \
           window[i] < window[i+1] and window[i] < window[i+2]:
            troughs.append(i)
    
    # Need at least 2 peaks and 1 trough for bullish impulse, or vice versa
    if len(peaks) >= 2 and len(troughs) >= 1:
        # Check for bullish impulse: trough -> peak -> trough -> peak (wave 1-2-3-4)
        # or peak -> trough -> peak -> trough -> peak (extended 3rd wave)
        sorted_points = sorted(peaks + troughs)
        if len(sorted_points) >= 4:
            # Simple momentum check: recent price action showing strength
            recent_trend = window[-1] - window[-10] if len(window) >= 10 else window[-1] - window[0]
            if recent_trend > 0 and len(peaks) >= len(troughs):
                return 1  # Bullish bias
            elif recent_trend < 0 and len(troughs) >= len(peaks):
                return -1  # Bearish bias
    
    return 0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ADX for trend strength (from 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    adx_1d = _calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Elliott Wave signal on 6h data
    ew_signal = np.zeros(n, dtype=int)
    for i in range(50, n):
        ew_signal[i] = _elliott_wave_signal(close[:i+1], lookback=30)
    
    # Volume confirmation (20-period average)
    volume = prices['volume'].values
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        # Trend regime from ADX
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: trend weakness or opposing wave signal
            if not is_trending or ew_signal[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: trend weakness or opposing wave signal
            if not is_trending or ew_signal[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # In trending markets: follow Elliott Wave impulse
            if is_trending and vol_confirm:
                if ew_signal[i] == 1:  # Bullish impulse
                    position = 1
                    signals[i] = 0.25
                elif ew_signal[i] == -1:  # Bearish impulse
                    position = -1
                    signals[i] = -0.25
            # In ranging markets: fade extreme wave corrections (optional)
            elif is_ranging and vol_confirm:
                # Could add mean reversion here, but keeping simple for now
                pass
    
    return signals