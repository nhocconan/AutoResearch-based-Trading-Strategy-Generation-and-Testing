#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w ADX regime filter and volume spike confirmation.
- Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13 (using 1d EMA13)
- Regime filter: 1w ADX > 25 for trending markets (strong trend), ADX < 20 for ranging markets
- In trending regime (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
- In ranging regime (ADX < 20): fade extremes - long when Bear Power < -0.5*ATR and turning up, short when Bull Power > 0.5*ATR and turning down
- Volume confirmation: > 1.5x 20-bar average to avoid false signals
- Designed for 6h timeframe to capture both trending and ranging market behavior
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn
- Targets 12-30 trades/year (50-120 total over 4 years) to stay fee-efficient
- Combines proven concepts: Elder Ray power indicators + ADX regime filtering + volume confirmation
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
    
    # Get 1d data ONCE before loop for Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d ATR for volatility normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1w ADX calculation for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])
    
    # Directional Movement
    dm_plus_1w = np.where(
        (high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]),
        np.maximum(high_1w[1:] - high_1w[:-1], 0),
        0
    )
    dm_minus_1w = np.where(
        (low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]),
        np.maximum(low_1w[:-1] - low_1w[1:], 0),
        0
    )
    dm_plus_1w = np.concatenate([[0], dm_plus_1w])
    dm_minus_1w = np.concatenate([[0], dm_minus_1w])
    
    # Smoothed TR, DM+, DM-
    tr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14_1w = pd.Series(dm_plus_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14_1w = pd.Series(dm_minus_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus_1w = 100 * dm_plus_14_1w / tr_14_1w
    di_minus_1w = 100 * dm_minus_14_1w / tr_14_1w
    
    # DX and ADX
    dx_1w = 100 * np.abs(di_plus_1w - di_minus_1w) / (di_plus_1w + di_minus_1w)
    adx_1w = pd.Series(dx_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime determination
        adx = adx_1w_aligned[i]
        is_trending = adx > 25
        is_ranging = adx < 20
        
        if position == 0:
            if is_trending and volume_confirm:
                # Trending regime: follow Elder Ray power
                bull_power = bull_power_1d_aligned[i]
                bear_power = bear_power_1d_aligned[i]
                
                # Long when Bull Power > 0 and increasing
                if bull_power > 0 and i > start_idx and bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short when Bear Power < 0 and decreasing
                elif bear_power < 0 and i > start_idx and bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging and volume_confirm:
                # Ranging regime: fade extremes
                bull_power = bull_power_1d_aligned[i]
                bear_power = bear_power_1d_aligned[i]
                atr = atr_1d_aligned[i]
                
                # Long when Bear Power is very negative and turning up
                if bear_power < -0.5 * atr and i > start_idx and bear_power_1d_aligned[i] > bear_power_1d_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short when Bull Power is very positive and turning down
                elif bull_power > 0.5 * atr and i > start_idx and bull_power_1d_aligned[i] < bull_power_1d_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # Exit when Bull Power turns negative
                if bull_power_1d_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging
                # Exit when Bear Power reaches zero (mean reversion complete)
                if bear_power_1d_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # Exit when Bear Power turns positive
                if bear_power_1d_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging
                # Exit when Bull Power reaches zero (mean reversion complete)
                if bull_power_1d_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0