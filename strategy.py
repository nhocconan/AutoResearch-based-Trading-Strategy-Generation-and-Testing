#!/usr/bin/env python3
# 12h_1dWilderATR_Multiplier_Trend_Follow
# Hypothesis: Wilder's ATR-based trend following with 12h price action and 1d trend filter (EMA100) works in both bull and bear markets by capturing strong trends while avoiding chop. Uses ATR multiplier for dynamic stop and entry. Low trade frequency (target: 15-30/year) minimizes fee drift. Uses 1d HTF for trend and ATR-based signals.

name = "12h_1dWilderATR_Multiplier_Trend_Follow"
timeframe = "12h"
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
    
    # Get daily data for trend filter and ATR-based calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA100 for trend filter
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Calculate daily Wilder's ATR (14-period) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's ATR: smoothed with alpha = 1/period
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[13] = np.mean(tr[1:14])  # First ATR value is average of first 14 TR
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate dynamic entry bands using ATR multiplier
    # Upper band: EMA100 + 2.5 * ATR
    # Lower band: EMA100 - 2.5 * ATR
    upper_band = ema_100_1d + 2.5 * atr_14_1d
    lower_band = ema_100_1d - 2.5 * atr_14_1d
    
    # Align all indicators to 12h timeframe
    ema_100_12h = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    upper_band_12h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_12h = align_htf_to_ltf(prices, df_1d, lower_band)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_100_12h[i]) or np.isnan(upper_band_12h[i]) or 
            np.isnan(lower_band_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above upper band with uptrend
            if close[i] > upper_band_12h[i] and close[i] > ema_100_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below lower band with downtrend
            elif close[i] < lower_band_12h[i] and close[i] < ema_100_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below EMA100 (trend change) or below lower band (reversal)
            if close[i] < ema_100_12h[i] or close[i] < lower_band_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above EMA100 (trend change) or above upper band (reversal)
            if close[i] > ema_100_12h[i] or close[i] > upper_band_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals