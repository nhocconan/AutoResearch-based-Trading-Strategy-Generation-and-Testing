#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume confirmation.
- Uses 6h timeframe (primary) and 1d HTF for ADX trend strength and EMA21 for Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA21(1d), Bear Power = Low - EMA21(1d). Measures buying/selling pressure relative to trend.
- Regime filter: Only trade when 1d ADX > 25 (trending market). In ranging markets (ADX < 20), stay flat.
- Entry logic: Long when Bull Power > 0 AND Bear Power < previous Bear Power (bullish momentum building) AND volume spike.
               Short when Bear Power < 0 AND Bull Power < previous Bull Power (bearish momentum building) AND volume spike.
- Volume confirmation: current 6h volume > 1.5 * 20-period 6h volume MA (moderate to avoid overtrading).
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull/bear: ADX filter ensures we only trade in trending conditions, Elder Ray captures momentum shifts.
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
    
    # Calculate 1d EMA21 for Elder Ray and 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA21 for Elder Ray calculation
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # 1d ADX calculation (trend strength indicator)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Same length as close_1d
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: prev*(period-1)/period + current/period
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align 1d indicators to 6h timeframe
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray calculation: Bull Power = High - EMA21, Bear Power = Low - EMA21
    bull_power = high - ema_21_1d_aligned
    bear_power = low - ema_21_1d_aligned
    
    # Previous Bear Power and Bull Power for momentum confirmation
    prev_bear_power = np.roll(bear_power, 1)
    prev_bull_power = np.roll(bull_power, 1)
    prev_bear_power[0] = np.nan
    prev_bull_power[0] = np.nan
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Regime filter: ADX > 25 for trending market
    trending = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30)  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(prev_bull_power[i]) or np.isnan(prev_bear_power[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (above EMA21) AND Bear Power decreasing (momentum building) AND volume spike AND trending
            if bull_power[i] > 0 and bear_power[i] < prev_bear_power[i] and volume_spike[i] and trending[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (below EMA21) AND Bull Power decreasing (momentum building) AND volume spike AND trending
            elif bear_power[i] < 0 and bull_power[i] < prev_bull_power[i] and volume_spike[i] and trending[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative (below EMA21) or Bear Power starts increasing (momentum weakening)
            if bull_power[i] <= 0 or bear_power[i] >= prev_bear_power[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive (above EMA21) or Bull Power starts increasing (momentum weakening)
            if bear_power[i] >= 0 or bull_power[i] >= prev_bull_power[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0