#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d ADX trend filter + volume confirmation
# Uses Donchian channel from 6h timeframe for breakout detection
# Enters long when price breaks above upper Donchian with volume > 1.5 x 20-period EMA and bullish 1d trend (ADX > 25 and +DI > -DI)
# Enters short when price breaks below lower Donchian with volume > 1.5 x 20-period EMA and bearish 1d trend (ADX > 25 and -DI > +DI)
# Exits on opposite Donchian breakout or when 1d trend weakens (ADX < 20)
# Volume spike confirms institutional participation, reducing false breakouts
# ADX filter ensures we only trade in trending markets, avoiding choppy conditions
# Designed for 6h timeframe targeting 12-37 trades/year with discrete sizing (0.25)

name = "6h_Donchian20_1dADXTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (+DI, -DI)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
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
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * dm_plus_smooth / tr_smooth
    minus_di = 100 * dm_minus_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = np.full_like(dx, np.nan)
    # First ADX value is simple average of first 'period' DX values
    if len(dx) >= 2*period:
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        # Wilder smoothing for ADX
        for i in range(2*period, len(dx)):
            if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX, +DI, -DI to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Get 6h data for Donchian channel (20-period)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian upper and lower bands
    donchian_upper = np.full_like(high_6h, np.nan)
    donchian_lower = np.full_like(low_6h, np.nan)
    
    for i in range(len(high_6h)):
        if i >= 19:  # 20-period lookback
            donchian_upper[i] = np.max(high_6h[i-19:i+1])
            donchian_lower[i] = np.min(low_6h[i-19:i+1])
    
    # Align Donchian bands to 6h timeframe (already aligned since we used 6h data)
    donchian_upper_aligned = donchian_upper
    donchian_lower_aligned = donchian_lower
    
    # Get 6h data for volume EMA(20) for volume confirmation
    if len(df_6h) < 20:
        return np.zeros(n)
    
    vol_6h = df_6h['volume'].values
    vol_ema_20 = pd.Series(vol_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = vol_ema_20  # Already aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        # 1d trend conditions
        bullish_trend = (adx_aligned[i] > 25) and (plus_di_aligned[i] > minus_di_aligned[i])
        bearish_trend = (adx_aligned[i] > 25) and (minus_di_aligned[i] > plus_di_aligned[i])
        weak_trend = adx_aligned[i] < 20  # Trend weakening
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + bullish 1d trend
            if (close[i] > donchian_upper_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + bearish 1d trend
            elif (close[i] < donchian_lower_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian OR trend weakens
            if close[i] < donchian_lower_aligned[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above upper Donchian OR trend weakens
            if close[i] > donchian_upper_aligned[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals