#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ADXTrend
Hypothesis: Donchian(20) breakout with volume spike and ADX trend filter on 4h.
Buy when price breaks above upper band with volume spike and ADX>25 (trending up).
Sell when price breaks below lower band with volume spike and ADX>25 (trending down).
Designed for low trade frequency (20-50/year) to avoid fee drag while capturing
trend momentum in both bull and bear markets. ADX filter avoids range-bound whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for ADX calculation (trend filter)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nansum(data[1:period]) if not np.all(np.isnan(data[1:period])) else np.nan
                # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
                for i in range(period, len(data)):
                    if np.isnan(result[i-1]) or np.isnan(data[i]):
                        result[i] = np.nan
                    else:
                        result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = wilder_smooth(dx, period)
        
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Donchian channels (20-period) on 4h
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    dc_upper_4h, dc_lower_4h = donchian_channels(high_4h, low_4h, 20)
    dc_upper_aligned = align_htf_to_ltf(prices, df_4h, dc_upper_4h)
    dc_lower_aligned = align_htf_to_ltf(prices, df_4h, dc_lower_4h)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(dc_upper_aligned[i]) or 
            np.isnan(dc_lower_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = dc_upper_aligned[i]
        lower = dc_lower_aligned[i]
        adx = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and ADX>25 (uptrend)
            if price > upper and vol_spike and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume spike and ADX>25 (downtrend)
            elif price < lower and vol_spike and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns below upper Donchian OR ADX drops below 20 (trend weakening)
            if price < upper or adx < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns above lower Donchian OR ADX drops below 20 (trend weakening)
            if price > lower or adx < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0