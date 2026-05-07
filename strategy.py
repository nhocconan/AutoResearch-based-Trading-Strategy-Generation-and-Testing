#!/usr/bin/env python3
name = "6h_ADX_Alligator_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Alligator lines: Jaw (13-period, 8-shift), Teeth (8-period, 5-shift), Lips (5-period, 3-shift)
    jaw = pd.Series(high_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(low_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # ADX on 1d for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values (14-period Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    # DI and DX
    plus_di = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
            bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
            
            vol_condition = volume[i] > vol_ma_24[i] * 1.8
            strong_trend = adx_aligned[i] > 25
            
            # Long: bullish alignment + volume + strong trend
            if bullish_alignment and vol_condition and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + volume + strong trend
            elif bearish_alignment and vol_condition and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: alignment breaks or ADX weakens
            if not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: alignment breaks or ADX weakens
            if not (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Williams Alligator + ADX trend filter with volume confirmation
# - Alligator identifies trend direction via jaw/teeth/lips alignment
# - ADX > 25 ensures strong trend, < 20 signals weakening for exit
# - Volume spike (1.8x average) confirms institutional participation
# - Works in bull markets (buy on bullish alignment) and bear markets (sell on bearish alignment)
# - Exit when alignment breaks or trend weakens
# - Position size 0.25 targets 20-40 trades/year, avoiding fee drag
# - Combines trend-following (Alligator) with trend-strength filter (ADX) for robustness
# - Volume filter reduces false signals during low-participation periods