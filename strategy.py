#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX regime + volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
- Regime filter: ADX(14) from 1d timeframe (trending if >25, ranging if <20)
- In trending regime (ADX>25): trend follow - long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
- In ranging regime (ADX<20): mean revert - long when Bull Power < -0.5*ATR and turning up, short when Bear Power < -0.5*ATR and turning down
- Volume confirmation: volume > 1.5 * median volume of last 20 bars
- Uses 6h primary timeframe with 1d HTF for regime to target 50-150 total trades over 4 years (12-37/year)
- Elder Ray measures price relative to EMA13 to show bull/bear power
- 1d ADX regime avoids whipsaws by adapting strategy to market conditions
- Volume confirmation reduces fakeouts
- Designed to work in both bull (trend following) and bear (mean revert in ranges) markets
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
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Calculate ATR(10) for position sizing in ranging regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = tr2_1d[0] = tr3_1d[0] = np.nan
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 13) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr10[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        vol_ok = volume_confirm[i]
        
        if position == 0:
            # Determine regime
            if adx_val > 25:  # Trending regime
                # Trend following: long when bull power positive and rising
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short when bear power positive and rising
                elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and vol_ok:
                    signals[i] = -0.25
                    position = -1
            elif adx_val < 20:  # Ranging regime
                # Mean reversion: long when bull power negative but turning up
                if bull_power[i] < -0.5 * atr10[i] and bull_power[i] > bull_power[i-1] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short when bear power negative but turning down
                elif bear_power[i] < -0.5 * atr10[i] and bear_power[i] > bear_power[i-1] and vol_ok:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit conditions
            if adx_val > 25:  # Trending: exit when bull power turns negative
                if bull_power[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit when bull power reaches zero or turns down
                if bull_power[i] >= 0 or bull_power[i] < bull_power[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if adx_val > 25:  # Trending: exit when bear power turns negative
                if bear_power[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit when bear power reaches zero or turns up
                if bear_power[i] >= 0 or bear_power[i] > bear_power[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0