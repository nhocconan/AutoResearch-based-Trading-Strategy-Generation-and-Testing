#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation
- Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength
- ADX > 25 indicates trending market (use Elder Ray signals), ADX < 20 indicates ranging (fade extremes)
- Volume confirmation (> 1.3x 20-period average) ensures momentum behind moves
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in bull markets via Bull Power > 0 + uptrend, bear markets via Bear Power > 0 + downtrend
- In ranging markets, fades when Bull/Bear Power exceeds 2 std dev from mean
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA(13) for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema_13_aligned
    bear_power = ema_13_aligned - low
    
    # Calculate 1d ADX(14) for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    tr_smooth = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    tr_smooth[tr_period-1] = np.nansum(tr[:tr_period])
    dm_plus_smooth[tr_period-1] = np.nansum(dm_plus[:tr_period])
    dm_minus_smooth[tr_period-1] = np.nansum(dm_minus[:tr_period])
    
    for i in range(tr_period, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / tr_period) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.zeros_like(di_plus)
    mask = (di_plus + di_minus) != 0
    dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / (di_plus[mask] + di_minus[mask])
    
    adx_period = 14
    adx = np.zeros_like(dx)
    if len(dx) >= adx_period:
        adx[adx_period-1] = np.nanmean(dx[:adx_period])
        for i in range(adx_period, len(dx)):
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 14+13, 20)  # EMA, ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        # Calculate rolling mean/std for Elder Ray power (for mean reversion in ranging)
        if i >= 50:
            lookback = min(50, i)
            bp_mean = np.nanmean(bull_power[i-lookback:i])
            bp_std = np.nanstd(bull_power[i-lookback:i]) if np.nanstd(bull_power[i-lookback:i]) > 0 else 1
            br_mean = np.nanmean(bear_power[i-lookback:i])
            br_std = np.nanstd(bear_power[i-lookback:i]) if np.nanstd(bear_power[i-lookback:i]) > 0 else 1
            
            bp_z = (bull_power[i] - bp_mean) / bp_std
            br_z = (bear_power[i] - br_mean) / br_std
        else:
            bp_z = 0
            br_z = 0
        
        if position == 0:
            # Long conditions
            long_signal = False
            if trending:
                # In trending market: bull power positive and rising
                long_signal = (bull_power[i] > 0 and 
                              bull_power[i] > bull_power[i-1] and
                              close[i] > close[i-1])
            elif ranging:
                # In ranging market: fade extreme bear power (oversold)
                long_signal = (br_z < -2.0 and  # extreme bear power
                              volume[i] > 1.3 * vol_ma[i])
            
            # Short conditions
            short_signal = False
            if trending:
                # In trending market: bear power positive and rising
                short_signal = (bear_power[i] > 0 and 
                               bear_power[i] > bear_power[i-1] and
                               close[i] < close[i-1])
            elif ranging:
                # In ranging market: fade extreme bull power (overbought)
                short_signal = (bp_z > 2.0 and   # extreme bull power
                               volume[i] > 1.3 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: bear power becomes positive OR bull power turns negative
                if (bear_power[i] > 0 and 
                    bear_power[i] > bear_power[i-1]) or bull_power[i] < 0:
                    exit_signal = True
                # Also exit if volatility expands sharply (potential reversal)
                elif i >= 20 and atr_ratio(i, high, low, close) > 2.5:
                    exit_signal = True
            elif position == -1:
                # Exit short: bull power becomes positive OR bear power turns negative
                if (bull_power[i] > 0 and 
                    bull_power[i] > bull_power[i-1]) or bear_power[i] < 0:
                    exit_signal = True
                # Also exit if volatility expands sharply (potential reversal)
                elif i >= 20 and atr_ratio(i, high, low, close) > 2.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

def atr_ratio(idx, high, low, close):
    """Calculate current ATR vs 20-period average ATR"""
    if idx < 20:
        return 1.0
    tr = np.zeros(20)
    for i in range(20):
        lookback = idx - i
        if lookback > 0:
            tr[i] = max(high[lookback] - low[lookback],
                       abs(high[lookback] - close[lookback-1]),
                       abs(low[lookback] - close[lookback-1]))
        else:
            tr[i] = high[lookback] - low[lookback]
    atr_now = np.mean(tr)
    
    # 20-period average ATR starting 20 periods ago
    if idx >= 40:
        tr_avg = np.zeros(20)
        for i in range(20):
            lookback = idx - 20 - i
            if lookback > 0:
                tr_avg[i] = max(high[lookback] - low[lookback],
                               abs(high[lookback] - close[lookback-1]),
                               abs(low[lookback] - close[lookback-1]))
            else:
                tr_avg[i] = high[lookback] - low[lookback]
        atr_avg = np.mean(tr_avg)
        if atr_avg > 0:
            return atr_now / atr_avg
    return 1.0

name = "6h_ElderRay_1dADXRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0