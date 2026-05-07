#!/usr/bin/env python3
# 4h_Camarilla_R1S1_1dATR10_Trend_Volume
# Hypothesis: Uses Camarilla pivot levels (R1/S1) on 1d chart with 1d ATR-based trend filter (close > EMA10 + ATR*0.5 for long, < EMA10 - ATR*0.5 for short) and volume confirmation. Designed to work in both bull and bear markets by trading only in the direction of the 1d ATR-adjusted trend. Target: 20-40 trades/year to minimize fee drag.

name = "4h_Camarilla_R1S1_1dATR10_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA10 and ATR10 for trend filter
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema_10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate True Range and ATR(10)
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate ATR-based trend: long if close > EMA10 + 0.5*ATR, short if close < EMA10 - 0.5*ATR
    trend_long_1d = close_1d > (ema_10_1d + 0.5 * atr_10_1d)
    trend_short_1d = close_1d < (ema_10_1d - 0.5 * atr_10_1d)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # Using previous day's data to avoid look-ahead
    high_shift = np.concatenate([[np.nan], high_1d[:-1]])
    low_shift = np.concatenate([[np.nan], low_1d[:-1]])
    close_shift = np.concatenate([[np.nan], close_1d[:-1]])
    
    camarilla_range = high_shift - low_shift
    r1 = close_shift + 1.1 * camarilla_range / 12
    s1 = close_shift - 1.1 * camarilla_range / 12
    
    # Align 1d indicators to 4h timeframe
    ema_10_1d_4h = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    atr_10_1d_4h = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    trend_long_1d_4h = align_htf_to_ltf(prices, df_1d, trend_long_1d.astype(float))
    trend_short_1d_4h = align_htf_to_ltf(prices, df_1d, trend_short_1d.astype(float))
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate volume spike on 4h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_10_1d_4h[i]) or np.isnan(atr_10_1d_4h[i]) or 
            np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(trend_long_1d_4h[i]) or np.isnan(trend_short_1d_4h[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 + uptrend filter + volume spike
            if close[i] > r1_4h[i] and trend_long_1d_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 + downtrend filter + volume spike
            elif close[i] < s1_4h[i] and trend_short_1d_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S1 or trend turns bearish
            if close[i] < s1_4h[i] or not trend_long_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R1 or trend turns bullish
            if close[i] > r1_4h[i] or not trend_short_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals