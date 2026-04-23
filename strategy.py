#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Regime Filter
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h data)
- 1d ADX regime filter: ADX > 25 = trending (follow Elder Ray signals), ADX < 20 = ranging (fade Elder Ray extremes)
- Volume confirmation: > 1.5x 20-period average reduces false signals
- Designed for 6h timeframe to capture swing moves with controlled frequency (target: 12-37 trades/year)
- Works in both bull/bear via regime adaptation: trend follow in strong trends, mean revert in ranges
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Higher = stronger bullish pressure
    bear_power = low - ema_13   # Lower (more negative) = stronger bearish pressure
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr_1d = WilderSmooth(tr, period)
    dm_plus_smooth = WilderSmooth(dm_plus, period)
    dm_minus_smooth = WilderSmooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = WilderSmooth(dx, period)
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30 + period + period, 20)  # EMA13, ADX calculation, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entries
            if adx_1d_aligned[i] > 25:  # Trending regime - follow Elder Ray
                # Long: Bull Power rising AND above zero AND volume confirmation
                if (bull_power[i] > bull_power[i-1] and bull_power[i] > 0 and 
                    volume[i] > 1.5 * vol_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power falling AND below zero AND volume confirmation
                elif (bear_power[i] < bear_power[i-1] and bear_power[i] < 0 and 
                      volume[i] > 1.5 * vol_ma[i]):
                    signals[i] = -0.25
                    position = -1
            elif adx_1d_aligned[i] < 20:  # Ranging regime - fade extremes
                # Long: Bear Power extreme (very negative) AND turning up AND volume
                if (bear_power[i] < np.percentile(bear_power[max(0,i-50):i+1], 10) and  # Bottom 10%
                    bear_power[i] > bear_power[i-1] and 
                    volume[i] > 1.5 * vol_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power extreme (very high) AND turning down AND volume
                elif (bull_power[i] > np.percentile(bull_power[max(0,i-50):i+1], 90) and  # Top 10%
                      bull_power[i] < bull_power[i-1] and 
                      volume[i] > 1.5 * vol_ma[i]):
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit trending: Bear Power turns positive OR ADX weakens
                # Exit ranging: Bull Power falls from extreme OR ADX strengthens (trend emerging)
                if adx_1d_aligned[i] > 25:
                    if bear_power[i] > 0:  # Bear power turned positive
                        exit_signal = True
                else:  # Ranging or weak trend
                    if bear_power[i] > np.percentile(bear_power[max(0,i-50):i+1], 50):  # Above median
                        exit_signal = True
                        
            elif position == -1:  # Short position
                # Exit trending: Bull Power turns negative OR ADX weakens
                # Exit ranging: Bear Power rises from extreme OR ADX strengthens
                if adx_1d_aligned[i] > 25:
                    if bull_power[i] < 0:  # Bull power turned negative
                        exit_signal = True
                else:  # Ranging or weak trend
                    if bull_power[i] < np.percentile(bull_power[max(0,i-50):i+1], 50):  # Below median
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