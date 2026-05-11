#!/usr/bin/env python3
# 4h_1d_1w_Alligator_ElderRay_Trend_Volume
# Hypothesis: Combines Williams Alligator (trend direction) and Elder Ray (bull/bear power) on 1d/1w timeframes.
# Alligator: Jaw (13-smoothed), Teeth (8-smoothed), Lips (5-smoothed). 
#   Bullish: Lips > Teeth > Jaw (green alignment)
#   Bearish: Lips < Teeth < Jaw (red alignment)
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
#   Bullish when Bull Power > 0 and rising, Bearish when Bear Power < 0 and falling.
# Entry: Alligator alignment + Elder Ray power + volume surge on 4h.
# Exit: Opposite Alligator alignment or Elder Ray power divergence.
# Works in bull (rides trends) and bear (captures breakdowns) with trend filters.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_1d_1w_Alligator_ElderRay_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def wilders_smoothing(data, period):
    """Wilder's smoothing (same as RSI smoothing)"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    result[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def ema_series(data, period):
    """Exponential moving average"""
    return pd.Series(data).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data for Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 34 or len(df_1w) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Williams Alligator (SMMA) ---
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    jaw_1d = wilders_smoothing(df_1d['close'].values, 13)
    teeth_1d = wilders_smoothing(df_1d['close'].values, 8)
    lips_1d = wilders_smoothing(df_1d['close'].values, 5)
    
    # --- 1d Elder Ray ---
    ema13_1d = ema_series(df_1d['close'].values, 13)
    bull_power_1d = df_1d['high'].values - ema13_1d  # High - EMA13
    bear_power_1d = df_1d['low'].values - ema13_1d   # Low - EMA13
    
    # --- 1w Williams Alligator for higher timeframe trend ---
    jaw_1w = wilders_smoothing(df_1w['close'].values, 13)
    teeth_1w = wilders_smoothing(df_1w['close'].values, 8)
    lips_1w = wilders_smoothing(df_1w['close'].values, 5)
    
    # Align all indicators to 4h
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # --- Volume confirmation (2.0x 20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for longest indicator (1w Alligator needs ~34 bars)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or np.isnan(lips_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment (1d)
        lips_gt_teeth = lips_1d_aligned[i] > teeth_1d_aligned[i]
        teeth_gt_jaw = teeth_1d_aligned[i] > jaw_1d_aligned[i]
        bullish_alligator = lips_gt_teeth and teeth_gt_jaw  # Green: Lips > Teeth > Jaw
        
        lips_lt_teeth = lips_1d_aligned[i] < teeth_1d_aligned[i]
        teeth_lt_jaw = teeth_1d_aligned[i] < jaw_1d_aligned[i]
        bearish_alligator = lips_lt_teeth and teeth_lt_jaw  # Red: Lips < Teeth < Jaw
        
        # Elder Ray power (1d)
        bull_power_rising = bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1]
        bear_power_falling = bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1]
        
        # Higher timeframe trend (1w Alligator)
        lips_1w_gt_teeth_1w = lips_1w_aligned[i] > teeth_1w_aligned[i]
        teeth_1w_gt_jaw_1w = teeth_1w_aligned[i] > jaw_1w_aligned[i]
        bullish_1w_trend = lips_1w_gt_teeth_1w and teeth_1w_gt_jaw_1w
        
        lips_1w_lt_teeth_1w = lips_1w_aligned[i] < teeth_1w_aligned[i]
        teeth_1w_lt_jaw_1w = teeth_1w_aligned[i] < jaw_1w_aligned[i]
        bearish_1w_trend = lips_1w_lt_teeth_1w and teeth_1w_lt_jaw_1w
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Bullish Alligator + Bull Power rising + 1w bullish trend + volume surge
            if bullish_alligator and bull_power_rising and bullish_1w_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + Bear Power falling + 1w bearish trend + volume surge
            elif bearish_alligator and bear_power_falling and bearish_1w_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: Bearish Alligator OR Bear Power falling OR 1w trend turns bearish
                if bearish_alligator or bear_power_falling or bearish_1w_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Bullish Alligator OR Bull Power rising OR 1w trend turns bullish
                if bullish_alligator or bull_power_rising or bullish_1w_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals