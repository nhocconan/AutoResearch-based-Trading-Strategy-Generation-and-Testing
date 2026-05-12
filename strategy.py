#!/usr/bin/env python3
# 6h_Williams_Alligator_ADX_Trend
# Hypothesis: Combines Williams Alligator (Elder Ray components) with ADX to identify
# strong trending regimes. Long when jaw > teeth > lips (bullish alignment) and ADX > 25.
# Short when jaw < teeth < lips (bearish alignment) and ADX > 25.
# Uses weekly trend filter: only trade in direction of weekly EMA(34).
# Designed to avoid whipsaws in ranging markets and capture sustained trends in both bull and bear markets.
# Targets 15-30 trades/year to minimize fee drag.

name = "6h_Williams_Alligator_ADX_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: three smoothed SMAs
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    def smma(arr, period):
        # Smoothed Moving Average: similar to Wilder's smoothing
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition (jaw shifted 8, teeth 5, lips 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted beginnings with nan
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = np.nan
        tr2[0] = np.nan
        tr3[0] = np.nan
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = np.nan
        down_move[0] = np.nan
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
        def wilder_smooth(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan, dtype=float)
            result = np.full_like(arr, np.nan, dtype=float)
            # First value is simple average
            result[period-1] = np.nanmean(arr[:period])
            # Subsequent values: Wilder smoothing
            alpha = 1.0 / period
            for i in range(period, len(arr)):
                if np.isnan(result[i-1]):
                    result[i] = np.nanmean(arr[i-period+1:i+1])
                else:
                    result[i] = (1 - alpha) * result[i-1] + alpha * arr[i]
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, period)
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Weekly trend filter: EMA(34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(adx[i]) or 
            np.isnan(weekly_ema_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator alignment signals
        bullish_alignment = jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]
        bearish_alignment = jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]
        
        # Strong trend filter
        strong_trend = adx[i] > 25
        
        # Weekly trend filter
        above_weekly_ema = close[i] > weekly_ema_aligned[i]
        below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        if position == 0:
            # LONG: Bullish Alligator alignment + strong trend + above weekly EMA
            if bullish_alignment and strong_trend and above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment + strong trend + below weekly EMA
            elif bearish_alignment and strong_trend and below_weekly_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Any of: bearish alignment, weak trend, or price crosses below weekly EMA
            if bearish_alignment or not strong_trend or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Any of: bullish alignment, weak trend, or price crosses above weekly EMA
            if bullish_alignment or not strong_trend or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals