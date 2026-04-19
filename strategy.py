#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and ADX trend filter.
# Long when price breaks above Donchian high(20), volume > 1.5x average, and ADX > 25.
# Short when price breaks below Donchian low(20), volume > 1.5x average, and ADX > 25.
# Exit when price crosses back below/above Donchian mid-point.
# Uses 1d timeframe with weekly ADX for trend strength.
# Target: 15-25 trades/year per symbol to stay within frequency limits.
name = "1d_Donchian20_ADX_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels using daily data
    donchian_len = 20
    donchian_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate ADX on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
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
    atr_1w = wilder_smooth(tr, period)
    # Avoid division by zero
    atr_1w = np.where(atr_1w == 0, np.finfo(float).eps, atr_1w)
    plus_di_1w = 100 * wilder_smooth(plus_dm, period) / atr_1w
    minus_di_1w = 100 * wilder_smooth(minus_dm, period) / atr_1w
    dx_denom = plus_di_1w + minus_di_1w
    dx_denom = np.where(dx_denom == 0, np.finfo(float).eps, dx_denom)
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / dx_denom
    adx_1w = wilder_smooth(dx_1w, period)
    
    # Align ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Get daily average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_len, 20)  # Ensure Donchian and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        adx = adx_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above Donchian high, ADX > 25, volume confirmation
            if price > upper and adx > 25 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low, ADX > 25, volume confirmation
            elif price < lower and adx > 25 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian mid-point
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian mid-point
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals