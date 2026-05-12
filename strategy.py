#!/usr/bin/env python3

# 12h_1W_1D_Alligator_Filter_Volume
# Hypothesis: Williams Alligator on 1week defines primary trend (JAW/TEETH/LIPS alignment).
# Entry on 12h when price crosses LIPS (13-period SMMA) in trend direction with volume confirmation.
# Uses 1day ADX > 20 to filter ranging markets. Works in bull/bear by requiring trend alignment.
# Targets 15-25 trades/year on 12h timeframe.

name = "12h_1W_1D_Alligator_Filter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA) - used in Williams Alligator"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=np.float64)
    result = np.full_like(data, np.nan, dtype=np.float64)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1w data for Alligator trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Williams Alligator on 1week
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    median_price_1w = (df_1w['high'].values + df_1w['low'].values) / 2.0
    jaw = smma(median_price_1w, 13)
    teeth = smma(median_price_1w, 8)
    lips = smma(median_price_1w, 5)
    
    # Shift as per Alligator definition (future shift for visualization, but we use unshifted for crossover)
    # For trading, we use the unshifted SMMA values for crossover signals
    jaw_1w = jaw
    teeth_1w = teeth
    lips_1w = lips
    
    # Align to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_12h = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_12h = align_htf_to_ltf(prices, df_1w, lips_1w)

    # Calculate ADX on 1day for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or
            np.isnan(adx_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Alligator conditions: Lips above Teeth above Jaw = bullish alignment
        # Lips below Teeth below Jaw = bearish alignment
        bullish_alignment = (lips_12h[i] > teeth_12h[i]) and (teeth_12h[i] > jaw_12h[i])
        bearish_alignment = (lips_12h[i] < teeth_12h[i]) and (teeth_12h[i] < jaw_12h[i])
        
        # Trend filter: ADX > 20 indicates trending market
        strong_trend = adx_12h[i] > 20

        if position == 0:
            # LONG: Price crosses above Lips with bullish alignment and volume confirmation
            if (close[i] > lips_12h[i]) and bullish_alignment and strong_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below Lips with bearish alignment and volume confirmation
            elif (close[i] < lips_12h[i]) and bearish_alignment and strong_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Lips or trend alignment breaks
            if (close[i] < lips_12h[i]) or not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Lips or trend alignment breaks
            if (close[i] > lips_12h[i]) or not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals