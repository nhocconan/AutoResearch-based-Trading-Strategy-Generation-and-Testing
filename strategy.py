#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h ADX regime filter and volume confirmation.
- Primary timeframe: 6h for lower trade frequency and reduced fee drag.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 13-period EMA).
- ADX(14) from 12h timeframe: ADX > 25 indicates trending market (use Elder Ray signals),
  ADX < 20 indicates ranging market (fade extreme Elder Ray values).
- Volume: Current 6h volume > 1.5 * 20-period volume MA to confirm institutional participation.
- Entry Logic:
  * Trending (ADX > 25): Long when Bull Power > 0 AND rising, Short when Bear Power < 0 AND falling.
  * Ranging (ADX < 20): Long when Bear Power < -0.5 * ATR(10) (extreme oversold),
    Short when Bull Power > 0.5 * ATR(10) (extreme overbought).
- Exit: Opposite signal or loss of volume confirmation.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
This combines trend-following in strong markets with mean reversion in ranging markets,
using volume confirmation to avoid false signals. Works in both bull and bear regimes
by adapting to market conditions via ADX.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Calculate ATR(10) for ranging market thresholds
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14)
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    df_12h_close = df_12h['close'].values
    
    # True Range and Directional Movement
    tr_12h = np.maximum(
        df_12h_high - df_12h_low,
        np.maximum(
            np.abs(df_12h_high - np.roll(df_12h_close, 1)),
            np.abs(df_12h_low - np.roll(df_12h_close, 1))
        )
    )
    tr_12h[0] = 0
    
    dm_plus = np.where(
        (df_12h_high - np.roll(df_12h_high, 1)) > (np.roll(df_12h_low, 1) - df_12h_low),
        np.maximum(df_12h_high - np.roll(df_12h_high, 1), 0),
        0
    )
    dm_minus = np.where(
        (np.roll(df_12h_low, 1) - df_12h_low) > (df_12h_high - np.roll(df_12h_high, 1)),
        np.maximum(np.roll(df_12h_low, 1) - df_12h_low, 0),
        0
    )
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_12h + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_12h + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period 6h volume MA
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_6h)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period volume MA
    volume_confirmed = volume > (1.5 * vol_ma_6h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 13, 10)  # Volume MA, ADX, EMA13, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(volume_confirmed[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(atr10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        adx_val = adx_12h_aligned[i]
        vol_ok = volume_confirmed[i]
        
        if position == 0:
            # Entry logic based on ADX regime
            if adx_val > 25.0:  # Trending market
                # Long: Bull Power > 0 and rising (current > previous)
                if curr_bull > 0 and curr_bull > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 and falling (current < previous)
                elif curr_bear < 0 and curr_bear < bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
            elif adx_val < 20.0:  # Ranging market
                # Long: Extreme Bear Power (oversold)
                if curr_bear < -0.5 * atr10[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Extreme Bull Power (overbought)
                elif curr_bull > 0.5 * atr10[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Opposite signal or loss of volume confirmation
            if (adx_val > 25.0 and curr_bear < 0) or (adx_val < 20.0 and curr_bull > 0.5 * atr10[i]) or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Opposite signal or loss of volume confirmation
            if (adx_val > 25.0 and curr_bull > 0) or (adx_val < 20.0 and curr_bear < -0.5 * atr10[i]) or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hADX_Regime_VolumeConfirmed_v1"
timeframe = "6h"
leverage = 1.0