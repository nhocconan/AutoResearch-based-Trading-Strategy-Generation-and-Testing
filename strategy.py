#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d ADX Regime Filter + Volume Spike
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
In strong trends (ADX>25 on 1d), trade in direction of power with volume confirmation.
In ranging markets (ADX<20), fade extreme power reversals.
Uses 6h timeframe with 1d HTF for regime. Targets 50-150 total trades over 4 years (12-37/year).
Works in bull/bear via regime adaptation.
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
    
    # Get 1d data for ADX regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX on 1d for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
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
    
    # Smoothed TR, DM+, DM- (Wilder smoothing = EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        result[period-1] = np.nanmean(arr[:period])  # first value is average
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period_adx = 14
    tr_smooth = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, period_adx)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power on 6h
    bull_power = high - ema_13_6h  # High - EMA
    bear_power = ema_13_6h - low   # EMA - Low
    
    # 6h volume MA for confirmation
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 13, period_adx*2)  # 20 for vol MA, 13 for EMA, 28 for ADX stability
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(ema_13_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        adx_val = adx_aligned[i]
        ema_13 = ema_13_6h[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ma = vol_ma_20_6h[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        # Power normalization for comparison (relative to price)
        bull_norm = bull / curr_close
        bear_norm = bear / curr_close
        
        if position == 0:
            # Look for entry signals based on regime
            # Trending market (ADX > 25): trade in direction of stronger power
            if adx_val > 25:
                # Strong bull power + volume confirmation → long
                if bull_norm > bear_norm and bull_norm > 0.001 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                # Strong bear power + volume confirmation → short
                elif bear_norm > bull_norm and bear_norm > 0.001 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Ranging market (ADX < 20): fade extreme power reversals
            elif adx_val < 20:
                # Extreme bull power → expect reversal → short
                if bull_norm > 0.006 and volume_confirm:  # unusually strong bull power
                    signals[i] = -0.20  # smaller size for mean reversion
                    position = -1
                # Extreme bear power → expect reversal → long
                elif bear_norm > 0.006 and volume_confirm:  # unusually strong bear power
                    signals[i] = 0.20   # smaller size for mean reversion
                    position = 1
                else:
                    signals[i] = 0.0
            # Transition regime (20 <= ADX <= 25): no trade
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: power deteriorates OR ADX drops below 20 (trend ending) OR volume dies
            if (bull_norm < 0.0005 or adx_val < 20 or curr_volume < vol_ma * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: power deteriorates OR ADX drops below 20 (trend ending) OR volume dies
            if (bear_norm < 0.0005 or adx_val < 20 or curr_volume < vol_ma * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # smaller size for mean reversion shorts
    
    return signals

name = "6h_ElderRay_Power_ADXRegime_Volume"
timeframe = "6h"
leverage = 1.0