#!/usr/bin/env python3
# 4H_12H_Camarilla_R2_S2_Breakout_12hEMA50_Trend_Volume
# Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R2 level (from 12h) with 12h uptrend and volume confirmation.
# Short when price breaks below Camarilla S2 level (from 12h) with 12h downtrend and volume confirmation.
# Uses 12h trend filter to avoid counter-trend trades and Camarilla levels from 12h for precise entries.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years).
# Uses 2x tighter stop (Camarilla S2/R2) to reduce false breakouts and improve win rate.

name = "4H_12H_Camarilla_R2_S2_Breakout_12hEMA50_Trend_Volume"
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
    
    # Get 12h data for Camarilla levels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for 12h: R2, S2 based on previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    # Camarilla R2 = close + (range * 1.1/6)
    # Camarilla S2 = close - (range * 1.1/6)
    camarilla_r2 = close_12h + (range_12h * 1.1 / 6)
    camarilla_s2 = close_12h - (range_12h * 1.1 / 6)
    
    # 12h trend: EMA(50) on close
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_12h > ema_50
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align 12h indicators to 4h
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s2)
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_r2_aligned[i]) or np.isnan(camarilla_s2_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Camarilla R2 + 12h uptrend + volume confirmation
            if close[i] > camarilla_r2_aligned[i] and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S2 + 12h downtrend + volume confirmation
            elif close[i] < camarilla_s2_aligned[i] and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Camarilla S2 (reversal) or trend changes
            if close[i] < camarilla_s2_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Camarilla R2 (reversal) or trend changes
            if close[i] > camarilla_r2_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals