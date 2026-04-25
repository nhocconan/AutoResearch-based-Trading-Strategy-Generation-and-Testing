#!/usr/bin/env python3
"""
6h Elder Ray + Regime Filter (ADX) + Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
Combine with ADX regime filter (ADX>25 = trending, ADX<20 = range) to avoid whipsaw.
In trending markets: go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising.
In range markets: fade extremes (long when Bull Power < -threshold, short when Bear Power < -threshold).
Weekly timeframe filter ensures alignment with higher timeframe trend (price > weekly EMA34 = bullish bias).
Volume confirmation (>1.5x 20-period MA) filters low-quality signals.
Designed for 6h timeframe targeting 50-150 total trades over 4 years.
Works in both bull and bear markets via ADX regime adaptation and weekly trend filter.
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
    
    # Get 1d data for EMA13 (Elder Ray) and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for EMA13 and ADX
        return np.zeros(n)
    
    # Get 1w data for weekly EMA34 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # Need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = pd.Series(df_1d['close'])
    ema_13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[period-1:2*period-1])
        # Rest is Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    def wilders_smoothing_adx(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[period-1:2*period-1])
        # Rest is Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    adx_1d = wilders_smoothing_adx(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Elder Ray components (6h)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Higher = stronger bullish momentum
    bear_power = ema_13 - low   # Higher = stronger bearish momentum
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20, 13, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_13_1d_val = ema_13_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = range
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Weekly trend filter: price above/below weekly EMA34
        weekly_bullish = curr_close > ema_34_1w_val
        weekly_bearish = curr_close < ema_34_1w_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Entry logic based on regime
            if is_trending:
                # Trending market: follow Elder Ray momentum with weekly filter
                long_entry = (bull_val > 0) and weekly_bullish and volume_confirm
                short_entry = (bear_val > 0) and weekly_bearish and volume_confirm
            elif is_ranging:
                # Ranging market: fade Elder Ray extremes (contrarian)
                long_entry = (bull_val < -0.1) and weekly_bullish and volume_confirm  # Oversold bounce
                short_entry = (bear_val < -0.1) and weekly_bearish and volume_confirm  # Overbought reversal
            else:
                # ADX between 20-25: transition zone, no entries
                long_entry = False
                short_entry = False
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Elder Ray momentum weakens OR weekly trend turns bearish OR ADX drops (trend ending)
            if bull_val < 0 or not weekly_bullish or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Elder Ray momentum weakens OR weekly trend turns bullish OR ADX drops (trend ending)
            if bear_val < 0 or not weekly_bearish or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_WeeklyEMA34Filter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0