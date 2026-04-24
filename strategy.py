#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
- Regime filter: 1d ADX > 25 for trending markets (only trade with trend)
- In uptrend (ADX>25 + +DI > -DI): long when Bull Power > 0 and rising
- In downtrend (ADX>25 + -DI > +DI): short when Bear Power > 0 and rising
- Volume confirmation: current 6h volume > 1.5 * 20-period 6h volume MA
- Works in bull/bear: ADX regime filter avoids ranging markets, Elder Ray captures momentum
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h
    bear_power = ema_13_6h - low
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    tr_smooth = wilder_smooth(tr, period_adx)
    dm_plus_smooth = wilder_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilder_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilder_smooth(dx, period_adx)
    
    # Align 1d indicators to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    di_plus_1d_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_1d_aligned = align_htf_to_ltf(prices, df_1d, di_minus)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Rising power confirmation (current > previous)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    # Handle first bar
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, period_adx*2)  # Need sufficient data for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(di_plus_1d_aligned[i]) or 
            np.isnan(di_minus_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filters
        strong_trend = adx_1d_aligned[i] > 25
        uptrend_regime = strong_trend and (di_plus_1d_aligned[i] > di_minus_1d_aligned[i])
        downtrend_regime = strong_trend and (di_minus_1d_aligned[i] > di_plus_1d_aligned[i])
        
        if position == 0:
            # Long: uptrend regime AND Bull Power > 0 AND rising AND volume spike
            if uptrend_regime and bull_power[i] > 0 and bull_power_rising[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend regime AND Bear Power > 0 AND rising AND volume spike
            elif downtrend_regime and bear_power[i] > 0 and bear_power_rising[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 or regime changes to downtrend
            if bull_power[i] <= 0 or not (uptrend_regime and adx_1d_aligned[i] > 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 or regime changes to uptrend
            if bear_power[i] <= 0 or not (downtrend_regime and adx_1d_aligned[i] > 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0