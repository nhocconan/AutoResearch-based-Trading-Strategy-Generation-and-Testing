#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
- Regime filter: ADX > 25 for trending (use Elder Ray signals), ADX <= 25 for ranging (fade extremes)
- In trending regime: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
- In ranging regime: long when Bear Power < -std and turning up, short when Bull Power > +std and turning down
- Volume confirmation: current 6h volume > 1.5 * 20-period volume MA (moderate to avoid overtrading)
- Uses 6h timeframe (primary) and 1d HTF for ADX and EMA13 (proven regime edge)
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
- Works in both bull/bear: regime adaptation avoids wrong-sided trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
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
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and ADX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx[adx == 0] = np.nan  # Avoid division by zero
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray components
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 14)  # Need sufficient lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate standard deviation of Bear Power for regime thresholds (20-period)
        if i >= 20:
            bear_power_std = np.nanstd(bear_power[max(0, i-20):i+1])
            bull_power_std = np.nanstd(bull_power[max(0, i-20):i+1])
        else:
            bear_power_std = 1.0
            bull_power_std = 1.0
        
        if position == 0:
            # Determine regime: trending (ADX > 25) or ranging (ADX <= 25)
            if adx_aligned[i] > 25:  # Trending regime
                # Long: Bull Power > 0 and rising (current > previous)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 and falling (current < previous)
                elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging regime
                # Long: Bear Power < -std and turning up (current > previous)
                if bear_power[i] < -bull_power_std and bear_power[i] > bear_power[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power > +std and turning down (current < previous)
                elif bull_power[i] > bear_power_std and bull_power[i] < bull_power[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative or regime shifts to ranging with overextension
            if bull_power[i] <= 0 or (adx_aligned[i] <= 25 and bull_power[i] > 2 * bull_power_std):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive or regime shifts to ranging with overextension
            if bear_power[i] >= 0 or (adx_aligned[i] <= 25 and bear_power[i] < -2 * bear_power_std):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0