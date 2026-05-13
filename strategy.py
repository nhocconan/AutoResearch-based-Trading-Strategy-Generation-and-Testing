#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
# Long when Williams %R < -80 (oversold) in a ranging market (ADX < 25) with volume > 1.3x average.
# Short when Williams %R > -20 (overbought) in a ranging market (ADX < 25) with volume > 1.3x average.
# Exit when Williams %R returns to -50 (mean reversion target) or volume drops below 0.7x average.
# Uses discrete sizing 0.25 to target 12-37 trades/year on 6h timeframe.
# Williams %R identifies extreme short-term momentum; ADX filter ensures mean reversion only in ranging markets.
# Volume confirmation avoids false signals during low-liquidity periods. Works in both bull and bear markets
# by fading momentum exhaustion in range-bound conditions.

name = "6h_WilliamsR_MeanReversion_1dADX25_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

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
    
    # Williams %R period (14)
    lookback_wr = 14
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    wr = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    
    # Get 1d data for ADX and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX (14) on 1d
    # ADX requires +DI and -DI calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate average volume for confirmation (20-period on 6h)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback_wr, n):
        # Skip if any required data is NaN
        if (np.isnan(wr[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) in ranging market (ADX < 25) with volume spike
            if (wr[i] < -80 and 
                adx_1d_aligned[i] < 25 and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20) in ranging market (ADX < 25) with volume spike
            elif (wr[i] > -20 and 
                  adx_1d_aligned[i] < 25 and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -50 (mean reversion) OR volume drops below 0.7x average
            if (wr[i] >= -50 or 
                volume[i] < 0.7 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -50 (mean reversion) OR volume drops below 0.7x average
            if (wr[i] <= -50 or 
                volume[i] < 0.7 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals