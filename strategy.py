#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator strategy with 1w trend filter and 1d volume confirmation.
# Long when price > Alligator Jaw (13-period SMMA) with 1w EMA50 > EMA100 (bullish trend) and 1d volume > 1.5x 20-period average.
# Short when price < Alligator Jaw with 1w EMA50 < EMA100 (bearish trend) and 1d volume > 1.5x 20-period average.
# Exit on opposite Alligator Teeth (8-period SMMA).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Williams Alligator catches trends early; 1w EMA filter ensures alignment with higher timeframe trend; volume confirmation reduces false signals.
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe.

name = "12h_WilliamsAlligator_1wEMATrend_1dVolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Williams Alligator (LTF) ---
    # Jaw: 13-period SMMA of median price, smoothed 8 bars
    # Teeth: 8-period SMMA of median price, smoothed 5 bars
    # Lips: 5-period SMMA of median price, smoothed 3 bars
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Additional smoothing as per Alligator definition
    jaw = smma(jaw_raw, 8)   # Jaw smoothed 8 more
    teeth = smma(teeth_raw, 5) # Teeth smoothed 5 more
    lips = smma(lips_raw, 3)   # Lips smoothed 3 more
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA50 and EMA100 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    ema_bullish = ema_50_1w_aligned > ema_100_1w_aligned  # Bullish 1w trend
    ema_bearish = ema_50_1w_aligned < ema_100_1w_aligned  # Bearish 1w trend
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or
            np.isnan(ema_bullish[i]) or np.isnan(ema_bearish[i]) or
            np.isnan(volume_confirm_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Jaw + 1w EMA50 > EMA100 + 1d volume confirmation
            if (close[i] > jaw[i] and 
                ema_bullish[i] and 
                volume_confirm_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Jaw + 1w EMA50 < EMA100 + 1d volume confirmation
            elif (close[i] < jaw[i] and 
                  ema_bearish[i] and 
                  volume_confirm_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < Teeth
            if close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > Teeth
            if close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals