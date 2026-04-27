#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with daily ADX trend filter and volume confirmation.
Breakouts above/below Donchian(20) capture momentum, ADX(14) > 25 filters for trending regimes,
volume > 1.5x 20-period average confirms institutional participation. Designed to work in both bull and bear markets by capturing strong directional moves. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=np.float64)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed values
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    
    # Initial average
    atr[period-1] = np.mean(tr[:period])
    plus_dm_sum = np.sum(plus_dm[:period])
    minus_dm_sum = np.sum(minus_dm[:period])
    
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    # DX and ADX
    dx = np.zeros_like(high)
    adx = np.zeros_like(high)
    for i in range(period, len(high)):
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0
    
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX(14)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    adx_14 = calculate_adx(daily_high, daily_low, daily_close, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate daily volume MA(20)
    daily_volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_period = 20
    upper_channel = np.full_like(high, np.nan, dtype=np.float64)
    lower_channel = np.full_like(high, np.nan, dtype=np.float64)
    
    for i in range(donchian_period-1, len(high)):
        upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + ADX (34) + volume MA (20)
    start_idx = max(donchian_period-1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_aligned[i]
        
        # Current indicators
        upper = upper_channel[i]
        lower = lower_channel[i]
        adx = adx_14_aligned[i]
        
        # Filters
        adx_filter = adx > 25  # Trending market
        vol_filter = vol_now > 1.5 * vol_ma  # Volume spike
        
        if position == 0:
            # Long: price breaks above upper Donchian with trend and volume
            if price_now > upper and adx_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian with trend and volume
            elif price_now < lower and adx_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian or trend weakens
            if price_now < lower or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above upper Donchian or trend weakens
            if price_now > upper or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_ADX_Volume_Breakout"
timeframe = "4h"
leverage = 1.0