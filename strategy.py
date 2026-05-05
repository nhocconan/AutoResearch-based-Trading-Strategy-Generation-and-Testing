#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + 1d Williams %R + volume confirmation
# Long when: ADX > 25 (trending), Williams %R < -80 (oversold on 1d), volume > 1.5x 24-period MA
# Short when: ADX > 25 (trending), Williams %R > -20 (overbought on 1d), volume > 1.5x 24-period MA
# Exit when: ADX < 20 (no trend) or Williams %R crosses midline (-50) in opposite direction
# Uses ADX for trend strength, Williams %R for overextension, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ADX_1dWilliamsR_VolumeConfirm"
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
    
    # Calculate volume confirmation on 6h using 24-period MA (equivalent to 1d lookback)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (1.5 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R on 1d timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    if len(close_1d) >= 14:
        highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
        # Handle division by zero when highest_high == lowest_low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(len(close_1d), np.nan)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate ADX on 6h timeframe (standard 14-period)
    if len(close) >= 14:
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0  # First period has no previous close
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nansum(data[:period]) / period
                # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current_data
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]):
                        result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = wilders_smoothing(tr, 14)
        plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        # Handle division by zero
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        adx = wilders_smoothing(dx, 14)
    else:
        adx = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: ADX > 25 (trending), Williams %R < -80 (oversold), volume filter
            if (adx[i] > 25 and 
                williams_r_aligned[i] < -80 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: ADX > 25 (trending), Williams %R > -20 (overbought), volume filter
            elif (adx[i] > 25 and 
                  williams_r_aligned[i] > -20 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 (no trend) or Williams %R crosses above -50
            if (adx[i] < 20 or williams_r_aligned[i] > -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX < 20 (no trend) or Williams %R crosses below -50
            if (adx[i] < 20 or williams_r_aligned[i] < -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals