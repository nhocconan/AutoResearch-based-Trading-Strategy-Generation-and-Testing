#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA of close)
- Regime: 1d ADX > 25 = trending (follow Elder Ray signals), ADX < 20 = range (fade Elder Ray extremes)
- Long: Bull Power > 0 AND volume > 1.5x 20-period avg AND (ADX > 25 OR (ADX < 20 and Bull Power < -0.5*ATR))
- Short: Bear Power < 0 AND volume > 1.5x 20-period avg AND (ADX > 25 OR (ADX < 20 and Bear Power > 0.5*ATR))
- Exit: Opposite Elder Ray signal crosses zero OR ADX regime flip (trending->range or range->trending)
- Uses Elder Ray for momentum, ADX for regime, volume for conviction
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (follow Bull Power in uptrend) and bear (fade Bear Power extremes in range)
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for adaptive thresholds
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 13)  # Need 20 for volume MA, 14 for ATR, 13 for EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or
            np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime conditions
        adx = adx_1d_aligned[i]
        is_trending = adx > 25
        is_range = adx < 20
        
        if position == 0:
            # Long conditions
            if volume_confirm:
                if is_trending and bull_power[i] > 0:
                    signals[i] = 0.25
                    position = 1
                elif is_range and bull_power[i] < -0.5 * atr[i]:
                    signals[i] = 0.25
                    position = 1
            # Short conditions
            elif volume_confirm:
                if is_trending and bear_power[i] < 0:
                    signals[i] = -0.25
                    position = -1
                elif is_range and bear_power[i] > 0.5 * atr[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bear Power crosses above zero OR regime flip to ranging
            if bear_power[i] > 0 or (is_trending and not is_range and adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power crosses below zero OR regime flip to ranging
            if bull_power[i] < 0 or (is_trending and not is_range and adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0