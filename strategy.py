#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + ADX regime with volume confirmation
- Elder Ray (Bull/Bear Power) from 1d: measures trend strength via EMA13
- ADX from 12h: filters for trending markets (ADX > 25)
- Volume confirmation: > 1.5x 20-period average to avoid false signals
- Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume confirmation
- Short: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 AND volume confirmation
- Exit: Elder Ray divergence (Bull Power < 0 for longs, Bear Power > 0 for shorts) OR ADX < 20
- Position size: 0.25 (discrete level)
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
- Works in bull/bear via ADX regime filter and Elder Ray strength measurement
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema_13_1d   # Bear Power = Low - EMA13
    
    # Align Elder Ray to 6h timeframe (use prior completed 1d bar)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 12h ADX for trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # First value is simple average of first 'period' values
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
                else:
                    result[i] = result[i-1]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 6h timeframe (use prior completed 12h bar)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA13 (1d) needs ~34 6h bars, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Elder Ray signals
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume confirmation
            if bull_power > 0 and bear_power < 0 and adx_val > 25 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 AND volume confirmation
            elif bear_power < 0 and bull_power > 0 and adx_val > 25 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power >= 0 (loss of bullish momentum) OR ADX < 20 (trend weak)
            if bear_power >= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power <= 0 (loss of bearish momentum) OR ADX < 20 (trend weak)
            if bull_power <= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeSpike_Session"
timeframe = "6h"
leverage = 1.0