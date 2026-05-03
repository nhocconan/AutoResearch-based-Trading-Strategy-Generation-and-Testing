#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Regime + Volume Spike
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d ADX > 25 defines trending regime, < 20 defines ranging regime (hysteresis)
# In trending regime (ADX >= 25): trend follow Elder Ray signals
# In ranging regime (ADX <= 20): mean revert at Elder Ray extremes
# Volume confirmation (2.0x 20-period EMA) filters low-conviction moves
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.

name = "6h_ElderRay_1dADXRegime_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    def wilders_smoothing_dx(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < period:
            return result
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    adx_14 = wilders_smoothing_dx(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate EMA13 for Elder Ray (6h)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 20-period EMA on 6h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start from 13 to have valid EMA13 and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(ema13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Regime definition with hysteresis
        if adx_aligned[i] >= 25:
            regime = 'trending'  # Strong trend
        elif adx_aligned[i] <= 20:
            regime = 'ranging'   # Weak trend/ranging
        else:
            regime = 'transition'  # Between 20-25, hold previous regime
            # For simplicity, we'll use the previous bar's regime or default to ranging
            if i > 0:
                regime = 'trending' if adx_aligned[i-1] >= 25 else 'ranging'
            else:
                regime = 'ranging'
        
        if position == 0:
            if regime == 'trending':
                # Trend following: follow Elder Ray direction
                if bull_power[i] > 0 and volume_spike:
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and volume_spike:
                    signals[i] = -0.25
                    position = -1
            else:  # ranging regime
                # Mean reversion: fade extreme Elder Ray readings
                if bull_power[i] > 0 and volume_spike:  # Overbought, short
                    signals[i] = -0.25
                    position = -1
                elif bear_power[i] < 0 and volume_spike:  # Oversold, long
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            # Exit long: Elder Ray turns bearish or regime changes against trend
            if regime == 'trending':
                # In trend: exit when bear power turns negative (trend weakening)
                if bear_power[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging
                # In range: exit when bull power normalizes (mean reversion complete)
                if bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Ray turns bullish or regime changes against trend
            if regime == 'trending':
                # In trend: exit when bull power turns positive (trend weakening)
                if bull_power[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging
                # In range: exit when bear power normalizes (mean reversion complete)
                if bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals