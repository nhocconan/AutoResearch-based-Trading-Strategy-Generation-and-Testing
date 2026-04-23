#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume spike confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d HTF)
- Regime: ADX > 25 = trending (follow Elder Ray signals), ADX < 20 = ranging (fade Elder Ray extremes)
- Entry: In trending regime, go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
- Volume confirmation: > 1.8x 20-period average to filter low-conviction moves
- Exit: Opposite Elder Ray signal OR ADX drops below 20 (regime shift to ranging)
- Designed for 6h timeframe to capture medium-term trends while avoiding 4h overtrading
- Works in bull markets via trend following, in bear markets via regime-adaptive fading
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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA13 for Elder Ray (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d High and Low for Elder Ray
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d ADX for regime filter (HTF = 1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for ADX, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Elder Ray signals
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        
        # Rising Bull Power: current > previous
        bull_power_rising = i > 0 and bull_power > bull_power_aligned[i-1]
        # Falling Bear Power: current < previous (more negative)
        bear_power_falling = i > 0 and bear_power < bear_power_aligned[i-1]
        
        # Regime filters
        adx = adx_1d_aligned[i]
        trending_regime = adx > 25
        ranging_regime = adx < 20
        
        if position == 0:
            # Long entry: Trending regime + Bull Power > 0 and rising + volume confirmation
            if trending_regime and bull_power > 0 and bull_power_rising and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Trending regime + Bear Power < 0 and falling + volume confirmation
            elif trending_regime and bear_power < 0 and bear_power_falling and volume_confirm:
                signals[i] = -0.25
                position = -1
            # Mean reversion in ranging regime: fade extreme Elder Ray readings
            elif ranging_regime and volume_confirm:
                if bull_power > np.percentile(bull_power_aligned[max(0, i-50):i+1], 80):
                    signals[i] = -0.20  # Short extreme bull power
                    position = -1
                elif bear_power < np.percentile(bear_power_aligned[max(0, i-50):i+1], 20):
                    signals[i] = 0.20   # Long extreme bear power
                    position = 1
        elif position == 1:
            # Long exit: Bear Power > 0 (trend weakness) OR ADX < 20 (regime shift to ranging)
            if bear_power > 0 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power < 0 (trend weakness) OR ADX < 20 (regime shift to ranging)
            if bull_power < 0 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0