#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume spike
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short) OR ADX < 20 (trend weak)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Williams %R captures momentum extremes, ADX filters for trending markets to avoid chop,
# volume spike confirms institutional participation. Works in bull markets via buying oversold dips in uptrends
# and bear markets via selling overbought rallies in downtrends.

name = "6h_WilliamsR_EXTREME_1dADX_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate Williams %R on 6h data (14-period)
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, -50)
    
    # Calculate ADX on 1d data (14-period)
    # ADX requires +DI, -DI, and TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values (Wilder's smoothing)
        def wilders_smoothing(values, period):
            result = np.full_like(values, np.nan)
            if len(values) >= period:
                # First value is simple average
                result[period-1] = np.nansum(values[:period]) / period
                # Subsequent values: smoothed = (prev_smoothed * (period-1) + current_value) / period
                for i in range(period, len(values)):
                    if not np.isnan(result[i-1]):
                        result[i] = (result[i-1] * (period-1) + values[i]) / period
            return result
        
        tr14 = wilders_smoothing(tr, 14)
        plus_dm14 = wilders_smoothing(plus_dm, 14)
        minus_dm14 = wilders_smoothing(minus_dm, 14)
        
        # Avoid division by zero
        plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
        minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
        
        dx = np.where((plus_di14 + minus_di14) != 0, 
                      np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
        adx = wilders_smoothing(dx, 14)
    else:
        adx = np.full(len(high_1d), np.nan)
    
    # Align Williams %R to 6h timeframe (no alignment needed as it's already 6h)
    williams_r_aligned = williams_r
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Trend filters: ADX > 25 indicates strong trend
    strong_trend = adx_aligned > 25
    weak_trend = adx_aligned < 20  # Exit when trend weakens
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND strong trend AND volume spike
            if (williams_r_aligned[i] < -80 and 
                strong_trend[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND strong trend AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  strong_trend[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 OR trend weakens
            if (williams_r_aligned[i] > -50 or 
                weak_trend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 OR trend weakens
            if (williams_r_aligned[i] < -50 or 
                weak_trend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals