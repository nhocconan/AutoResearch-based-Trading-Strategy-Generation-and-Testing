#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d ADX regime filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d ADX > 25 defines trending regime (only trade with trend), ADX < 20 defines ranging (fade extremes)
# Volume confirmation (1.5x 20-period EMA) filters low-conviction breakouts
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.
# Works in bull/bear via regime adaptation: trend follow in strong trends, mean revert in ranges.

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
    adx_regime = adx_14  # Using ADX(14), will apply thresholds 25 (trend) and 20 (range)
    
    # Align 1d ADX to 6h timeframe
    adx_regime_aligned = align_htf_to_ltf(prices, df_1d, adx_regime)
    
    # Calculate 6h EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: 20-period EMA on 6h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA and Elder Ray
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_regime_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Regime filters
        strong_trend = adx_regime_aligned[i] > 25   # Trending regime
        ranging_market = adx_regime_aligned[i] < 20  # Ranging regime
        
        if position == 0:
            # Long entry conditions
            if strong_trend and bull_power[i] > 0 and volume_spike:
                # Trend following: buy when bull power positive in strong uptrend
                signals[i] = 0.25
                position = 1
            elif ranging_market and bear_power[i] < 0 and volume_spike:
                # Mean reversion: buy when bear power negative (oversold) in ranging market
                signals[i] = 0.25
                position = 1
            # Short entry conditions
            elif strong_trend and bear_power[i] < 0 and volume_spike:
                # Trend following: sell when bear power negative in strong downtrend
                signals[i] = -0.25
                position = -1
            elif ranging_market and bull_power[i] > 0 and volume_spike:
                # Mean reversion: sell when bull power positive (overbought) in ranging market
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: power deteriorates or regime changes against position
            if (strong_trend and bull_power[i] <= 0) or (ranging_market and bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: power deteriorates or regime changes against position
            if (strong_trend and bear_power[i] >= 0) or (ranging_market and bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals