#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX trend filter.
# Long when price breaks above Donchian upper band, 1d volume > 1.5x average, and 1d ADX > 25.
# Short when price breaks below Donchian lower band, 1d volume > 1.5x average, and 1d ADX > 25.
# Exit when price crosses back below/above Donchian middle band (20-period average).
# Uses 4h timeframe with daily volume and ADX for confirmation.
# Target: 20-50 trades/year per symbol to stay within frequency limits.
name = "4h_Donchian20_1dVol_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 4h data
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle = (upper + lower) / 2.0
    
    # Calculate daily volume average
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1d = wilder_smooth(tr, period)
    # Avoid division by zero
    atr_1d = np.where(atr_1d == 0, np.finfo(float).eps, atr_1d)
    plus_di_1d = 100 * wilder_smooth(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, period) / atr_1d
    dx_denom = plus_di_1d + minus_di_1d
    dx_denom = np.where(dx_denom == 0, np.finfo(float).eps, dx_denom)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / dx_denom
    adx_1d = wilder_smooth(dx_1d, period)
    
    # Align daily volume MA and ADX to 4h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 30)  # Ensure Donchian and ADX are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        up = upper[i]
        low = lower[i]
        mid = middle[i]
        vol_ma = vol_ma_aligned[i]
        adx = adx_1d_aligned[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above upper band, ADX > 25, volume confirmation
            if price > up and adx > 25 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band, ADX > 25, volume confirmation
            elif price < low and adx > 25 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below middle band
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above middle band
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals