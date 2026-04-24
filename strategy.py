#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + ADX Regime with Volume Confirmation
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
- ADX > 25 indicates trending market (use EMA13 trend direction)
- ADX < 20 indicates ranging market (fade Elder Ray extremes)
- Volume > 1.5 * 20-bar median volume confirms participation
- In trend (ADX>25): Long when Bull Power > 0 and rising, Short when Bear Power > 0 and rising
- In range (ADX<20): Long when Bear Power < -0.5*ATR (oversold), Short when Bull Power < -0.5*ATR (overbought)
- Uses 6h primary timeframe with 1d HTF for EMA13/ATR/ADX to target 50-150 total trades over 4 years (12-37/year)
- Elder Ray measures bull/bear power relative to EMA, effective in both trending and ranging markets
- ADX regime filter adapts strategy to market conditions, reducing whipsaws
- Volume confirmation ensures breakouts/mean reversions have participation
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
    
    # Get 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ATR(14) for volatility normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ADX(14) for regime detection
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, np.nan)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, np.nan)
    
    tr_1d_for_dx = tr_1d  # reuse calculated TR
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / pd.Series(tr_1d_for_dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / pd.Series(tr_1d_for_dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_14_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Volume confirmation: volume > 1.5 * 20-bar median volume
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Elder Ray components for current bar
        bull_power = high[i] - ema_13_1d_aligned[i]
        bear_power = ema_13_1d_aligned[i] - low[i]
        
        if position == 0:
            # Regime-based entry logic
            if adx_14_1d_aligned[i] > 25:  # Trending market
                # Long: Bull Power positive and rising (vs previous bar)
                # Short: Bear Power positive and rising (vs previous bar)
                if i > start_idx:
                    prev_bull_power = high[i-1] - ema_13_1d_aligned[i-1]
                    prev_bear_power = ema_13_1d_aligned[i-1] - low[i-1]
                    if bull_power > 0 and bull_power > prev_bull_power and volume_confirm[i]:
                        signals[i] = 0.25
                        position = 1
                    elif bear_power > 0 and bear_power > prev_bear_power and volume_confirm[i]:
                        signals[i] = -0.25
                        position = -1
            elif adx_14_1d_aligned[i] < 20:  # Ranging market
                # Mean reversion at Elder Ray extremes
                # Long: Bear Power sufficiently negative (oversold)
                # Short: Bull Power sufficiently negative (overbought)
                if bull_power < -0.5 * atr_14_1d_aligned[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif bear_power < -0.5 * atr_14_1d_aligned[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit conditions
            if adx_14_1d_aligned[i] > 25:  # In trend: exit on trend weakness
                # Exit long if Bull Power turns negative
                if bull_power <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # In range: exit on mean reversion completion
                # Exit long if Elder Ray returns to neutral
                if bull_power >= -0.2 * atr_14_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if adx_14_1d_aligned[i] > 25:  # In trend: exit on trend weakness
                # Exit short if Bear Power turns negative
                if bear_power <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # In range: exit on mean reversion completion
                # Exit short if Elder Ray returns to neutral
                if bear_power >= -0.2 * atr_14_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0