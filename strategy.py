#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_V2
Hypothesis: Tighten entry conditions from prior version by requiring volume spike > 2.5x average (vs 2.0) and adding ADX(14) > 20 trend filter on 4h to avoid choppy markets. This reduces trade frequency while maintaining edge in trending markets. Camarilla R3/S3 breakouts with 1d EMA34 trend alignment capture institutional order flow. Volume spike confirms participation, ADX ensures we're not in ranging conditions. Designed for 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index with min_periods"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = np.nan
    dm_minus[0] = np.nan
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    tr_smoothed = WilderSmoothing(tr, period)
    dm_plus_smoothed = WilderSmoothing(dm_plus, period)
    dm_minus_smoothed = WilderSmoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = WilderSmoothing(dx, period)
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Camarilla levels (based on previous day's OHLC) - using R3/S3 (tighter bands for precision)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels: R3, S3 (standard bands)
    camarilla_range = 1.1 * (prev_high - prev_low)
    r3 = prev_close + camarilla_range * 0.25  # R3 level
    s3 = prev_close - camarilla_range * 0.25  # S3 level
    
    # Align Camarilla levels to 4h timeframe (already completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.5 * 20-period average (tighter than before)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    # ADX trend filter on 4h timeframe
    adx = calculate_adx(high, low, close, 14)
    strong_trend = adx > 20  # Only trade when ADX > 20 (trending market)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (34) + volume MA (20) + ADX (14*2) + Camarilla (2)
    start_idx = max(34, 20, 28, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla R3/S3 breakout + volume spike + 1d EMA34 trend alignment + ADX > 20
            long_breakout = curr_high > r3_aligned[i]
            short_breakout = curr_low < s3_aligned[i]
            
            long_entry = long_breakout and volume_spike[i] and (curr_close > ema_34_1d_aligned[i]) and strong_trend[i]
            short_entry = short_breakout and volume_spike[i] and (curr_close < ema_34_1d_aligned[i]) and strong_trend[i]
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below R3 (failed breakout) or trend turns bearish
            if curr_close < r3_aligned[i] or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: exit when price closes above S3 (failed breakout) or trend turns bullish
            if curr_close > s3_aligned[i] or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_V2"
timeframe = "4h"
leverage = 1.0