#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX Regime Filter + Volume Confirmation
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close)
- Regime: 1d ADX > 25 = trending (follow Elder Ray signals), ADX < 20 = ranging (fade Elder Ray extremes)
- Volume: Must be > 1.3 * ATR(14) * close to confirm momentum
- Long when: Bull Power > 0 AND (ADX > 25 OR (ADX < 20 and Bear Power < -0.5 * ATR)) AND volume confirmation
- Short when: Bear Power < 0 AND (ADX > 25 OR (ADX < 20 and Bull Power > 0.5 * ATR)) AND volume confirmation
- Exit when opposing Elder Ray power crosses zero (mean reversion in ranging, trend exhaustion in trending)
- Uses 6h primary timeframe with 1d HTF for regime detection to avoid false signals in chop
- Elder Ray measures price relative to EMA13, showing true bull/bear strength beyond simple price action
- ADX regime filter prevents trend-following whipsaws in ranging markets and fade failures in strong trends
- Volume confirmation ensures breakouts have conviction, reducing false signals
- Designed for BTC/ETH: works in bull trends (follow strength), bear trends (fade weakness), and ranges (mean revert at extremes)
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
    
    # Calculate EMA13 for Elder Ray (using 6h close)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    dm_plus.iloc[0] = np.nan
    dm_minus.iloc[0] = np.nan
    
    # Smoothed DM and TR
    dm_plus_smooth = dm_plus.ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = dm_minus.ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate ATR(14) for 6h timeframe (for volume threshold and exit conditions)
    tr1_6h = pd.Series(high - low)
    tr2_6h = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3_6h = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2_6h.iloc[0] = np.nan
    tr3_6h.iloc[0] = np.nan
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h = tr_6h.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.3 * ATR * close
    vol_threshold = 1.3 * atr_6h * close
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions
            # Trending market (ADX > 25): follow Elder Ray strength
            # Ranging market (ADX < 20): fade weakness (extreme Bear Power)
            trending_long = adx_aligned[i] > 25 and bull_power[i] > 0
            ranging_long = adx_aligned[i] < 20 and bear_power[i] < -0.5 * atr_6h[i]
            if (trending_long or ranging_long) and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions
            # Trending market: follow Elder Ray weakness
            # Ranging market: fade strength (extreme Bull Power)
            trending_short = adx_aligned[i] > 25 and bear_power[i] < 0
            ranging_short = adx_aligned[i] < 20 and bull_power[i] > 0.5 * atr_6h[i]
            if (trending_short or ranging_short) and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power crosses below zero (loss of bullish strength)
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power crosses above zero (loss of bearish strength)
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0