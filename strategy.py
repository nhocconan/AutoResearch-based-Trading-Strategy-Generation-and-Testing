#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d ADX25 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# Uses 6h primary timeframe with 1d HTF for ADX trend filter and Williams %R
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) based on proven Williams %R performance
# Works in both bull and bear markets by requiring trending conditions (ADX > 25) to avoid whipsaws in ranging markets

name = "6h_WilliamsR_Extreme_1dADX25_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data for trend filter
    # ADX calculation requires +DI, -DI, and DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = previous * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    def williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
        wr = (highest_high - close) / (highest_high - lowest_low) * -100
        # Handle division by zero
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)
        return wr.values
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(wr[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if (wr[i] < -80 and 
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif (wr[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (exiting oversold territory)
            if wr[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (exiting overbought territory)
            if wr[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals