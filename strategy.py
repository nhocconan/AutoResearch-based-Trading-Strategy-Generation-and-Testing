#!/usr/bin/env python3
# 6h_elder_ray_regime_v2
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
# Bull Power = EMA(13) - Low, Bear Power = High - EMA(13). Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with ADX > 25 (trending) and volume spike.
# Short when Bear Power < 0 and falling, Bull Power < 0 and rising, with ADX > 25 and volume spike.
# Uses 6h timeframe to reduce trade frequency. Elder Ray measures trend strength via price-EMA relationship.
# ADX regime filter ensures we only trade in strong trends (avoids chop). Volume confirms institutional participation.
# Designed for 12-37 trades/year (50-150 over 4 years) with discrete position sizing (0.25).
# Works in bull/bear markets: captures strong trends while avoiding false signals in ranging conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = EMA(13) - Low, Bear Power = High - EMA(13)
    bull_power = ema_13 - low
    bear_power = high - ema_13
    
    # Get 1d HTF data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe (completed daily candle only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Rising/falling power detection (1-bar momentum)
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        bull_falling = bull_power[i] < bull_power[i-1]
        bear_rising = bear_power[i] > bear_power[i-1]
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative OR ADX weakens (< 20) OR Bear Power becomes positive
            if (bull_power[i] <= 0) or (adx_aligned[i] < 20) or (bear_power[i] > 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR ADX weakens (< 20) OR Bull Power becomes negative
            if (bear_power[i] >= 0) or (adx_aligned[i] < 20) or (bull_power[i] < 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Bull Power > 0 and rising, Bear Power < 0 and falling, ADX > 25 (strong trend), volume spike
            if (bull_power[i] > 0 and bull_rising and bear_power[i] < 0 and bear_falling and 
                adx_aligned[i] > 25 and vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power < 0 and falling, Bull Power < 0 and rising, ADX > 25 (strong trend), volume spike
            elif (bear_power[i] < 0 and bear_falling and bull_power[i] < 0 and bull_rising and 
                  adx_aligned[i] > 25 and vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals