#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
Long when Bull Power > 0 AND 1d ADX > 25 (trending) AND volume > 1.5x average.
Short when Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x average.
Exit when Elder Ray power reverses sign OR 1d ADX drops below 20 (range regime).
Designed for low trade frequency (~15-25/year) to capture strong trending moves while avoiding whipsaws in ranging markets.
Works in both bull and bear markets by using ADX to filter for trending conditions only.
Elder Ray measures trend strength relative to EMA13, making it effective in volatile crypto markets.
"""

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
    
    # Load 1d data for ADX calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX for 1d timeframe (trend strength filter)
    # ADX calculation: +DM, -DM, TR, then smoothed to get +DI, -DI, DX, then ADX
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for first element
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values using Wilder's smoothing (alpha = 1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(data[:period])
                # Subsequent values: Wilder's smoothing
                for i in range(period, len(data)):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h timeframe
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: measures bullish strength
    bear_power = low - ema13   # Bear Power: measures bearish strength (negative values)
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum) AND ADX > 25 (strong trend) AND volume spike
            if (bull_val > 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish momentum) AND ADX > 25 (strong trend) AND volume spike
            elif (bear_val < 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power turns negative OR ADX drops below 20 (trend weakening)
                if bull_val <= 0 or adx_val < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bear Power turns positive OR ADX drops below 20 (trend weakening)
                if bear_val >= 0 or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dADX_Volume_Trend"
timeframe = "6h"
leverage = 1.0