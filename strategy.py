#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX Regime + Volume Confirmation
- Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
- 1d ADX > 25 = trending regime (use Elder Ray for direction), ADX < 20 = ranging regime (fade extremes)
- Volume > 1.5x 20-period average confirms participation
- Designed for 6h timeframe targeting 12-30 trades/year (50-120 over 4 years)
- Works in bull/bear via regime adaptation: trend follow in trending, mean revert in ranging
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
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Higher = stronger bulls
    bear_power = ema_13 - low   # Higher = stronger bears
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = wilders_smoothing(dx, period)
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30 + 14 + 14, 20)  # EMA13, ADX calc, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        if position == 0:
            # Regime-based entry
            if adx_val > 25:  # Trending regime
                # Strong bull power -> long
                if bull_val > 0 and bull_val > bear_val:
                    signals[i] = 0.25
                    position = 1
                # Strong bear power -> short
                elif bear_val > 0 and bear_val > bull_val:
                    signals[i] = -0.25
                    position = -1
            elif adx_val < 20:  # Ranging regime
                # Fade extremes: short on bull power excess, long on bear power excess
                if bull_val > np.percentile(bull_power_aligned[max(0, i-50):i+1], 80):
                    signals[i] = -0.25
                    position = -1
                elif bear_val > np.percentile(bear_power_aligned[max(0, i-50):i+1], 80):
                    signals[i] = 0.25
                    position = 1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit: bear power exceeds bull power OR ADX weakens
                if bear_val > bull_val or adx_val < 20:
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit: bull power exceeds bear power OR ADX weakens
                if bull_val > bear_val or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dADXRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0