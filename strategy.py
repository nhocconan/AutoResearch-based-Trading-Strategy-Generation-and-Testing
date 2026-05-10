#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: 4H breakout of Camarilla R1/S1 levels with 12H EMA50 trend filter and volume confirmation.
# The Camarilla pivot system identifies key intraday support/resistance levels.
# Combining with higher timeframe trend (12H) and volume confirmation filters false breakouts.
# Designed for 4H timeframe to target 20-50 trades/year, minimizing fee drag while capturing
# sustained moves in both bull and bear markets. Works by buying strength in uptrends and
# selling weakness in downtrends, avoiding counter-trend trades.

name = "4H_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12H data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Daily data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need to align these levels to the 4H chart
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate R1 and S1 for each day
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align daily Camarilla levels to 4H timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 12H trend: EMA50 on 12H close
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12H trend to 4H
    trend_12h_up_4h = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_4h = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Volume confirmation: 20-period average on 4H
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(trend_12h_up_4h[i]) or np.isnan(trend_12h_down_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: break above Camarilla R1 with 12H uptrend and volume
            if (close[i] > r1_4h[i] and 
                trend_12h_up_4h[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: break below Camarilla S1 with 12H downtrend and volume
            elif (close[i] < s1_4h[i] and 
                  trend_12h_down_4h[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to Camarilla S1 or trend fails
            if (close[i] < s1_4h[i] or 
                trend_12h_up_4h[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to Camarilla R1 or trend fails
            if (close[i] > r1_4h[i] or 
                trend_12h_down_4h[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals