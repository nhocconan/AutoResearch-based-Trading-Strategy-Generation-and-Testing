#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX Regime + Volume Confirmation
- Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
- Regime filter: 1d ADX(14) > 25 = trending, < 20 = ranging (hysteresis)
- In trending regime (ADX > 25): trend follow - long when Bull Power > 0, short when Bear Power < 0
- In ranging regime (ADX < 20): mean revert - long when Bear Power < -0.5*ATR, short when Bull Power > 0.5*ATR
- Volume confirmation: require volume > 1.5x 20-period average to avoid low-volatility false signals
- Designed to work in both bull and bear markets via regime adaptation
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
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
    
    # Calculate ATR(14) for regime-based thresholds
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Higher highs relative to trend
    bear_power = ema_13 - low   # Lower lows relative to trend
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    # Load 1d ADX(14) ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate ADX components
    plus_dm = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    minus_dm = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    tr_1d = np.maximum(np.abs(df_1d['high'] - df_1d['low']), 
                       np.maximum(np.abs(df_1d['high'] - df_1d['close'].shift(1)), 
                                  np.abs(df_1d['low'] - df_1d['close'].shift(1))))
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    tr_14 = wilders_smoothing(tr_1d, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = wilders_smoothing(dx_14, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 14)  # Need 20 for volume MA, 13 for EMA, 14 for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(volume_confirm[i]) or 
            np.isnan(adx_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime determination with hysteresis
        adx_val = adx_14_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            # No position - look for entry
            if is_trending:
                # Trending regime: trend following
                if bull_power[i] > 0 and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] > 0 and volume_confirm[i]:  # Note: bear_power > 0 means negative bear power
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Ranging regime: mean reversion
                if bear_power[i] < -0.5 * atr[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif bull_power[i] > 0.5 * atr[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position - look for exit
            exit_condition = False
            if is_trending:
                # Exit trend follow when bull power fades
                if bull_power[i] <= 0:
                    exit_condition = True
            elif is_ranging:
                # Exit mean reversion when price moves to midpoint
                if bear_power[i] > -0.2 * atr[i]:
                    exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position - look for exit
            exit_condition = False
            if is_trending:
                # Exit trend follow when bear power fades
                if bear_power[i] <= 0:
                    exit_condition = True
            elif is_ranging:
                # Exit mean reversion when price moves to midpoint
                if bull_power[i] < 0.2 * atr[i]:
                    exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0