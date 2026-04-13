#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d ADX regime filter
    # Long: Elder Bull Power > 0 AND 1d ADX > 25 (strong trend)
    # Short: Elder Bear Power < 0 AND 1d ADX > 25 (strong trend)
    # Exit: Elder Power crosses zero OR 1d ADX < 20 (weak trend/no trend)
    # Uses 6h for Elder Ray (trend strength), 1d for ADX (regime filter)
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for Elder Ray (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1d data for ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # EMA13 on 6h close
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power_6h = high_6h - ema13_6h
    bear_power_6h = low_6h - ema13_6h
    
    # Align 6h Elder Ray to 6h timeframe (no additional delay for price-based indicators)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    # Calculate 1d ADX (Average Directional Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing for TR, DM+, DM-
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(atr_1d, np.nan)
    di_minus = np.full_like(atr_1d, np.nan)
    mask = ~np.isnan(atr_1d) & (atr_1d > 0)
    di_plus[mask] = 100 * dm_plus_smoothed[mask] / atr_1d[mask]
    di_minus[mask] = 100 * dm_minus_smoothed[mask] / atr_1d[mask]
    
    # DX and ADX
    dx = np.full_like(atr_1d, np.nan)
    dx_mask = mask & ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) > 0)
    dx[dx_mask] = 100 * np.abs(di_plus[dx_mask] - di_minus[dx_mask]) / (di_plus[dx_mask] + di_minus[dx_mask])
    
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1d ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        # Exit regime: ADX < 20 (weak trend/no trend)
        weak_trend = adx_aligned[i] < 20
        
        # Elder Ray signals
        bull_signal = bull_power_aligned[i] > 0
        bear_signal = bear_power_aligned[i] < 0
        
        # Entry logic: Elder Ray signal + strong trend regime
        long_entry = bull_signal and strong_trend
        short_entry = bear_signal and strong_trend
        
        # Exit logic: Elder Power crosses zero OR regime shifts to weak trend
        long_exit = (bull_power_aligned[i] <= 0) or weak_trend
        short_exit = (bear_power_aligned[i] >= 0) or weak_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0