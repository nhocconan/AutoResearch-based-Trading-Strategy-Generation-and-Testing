#!/usr/bin/env python3
# 4h_1d_Camarilla_R3_S3_Breakout_Volume
# Hypothesis: Uses 1d Camarilla R3/S3 levels as breakout triggers with 4h timeframe execution.
# Long when price breaks above R3 with volume confirmation, short when breaks below S3.
# Uses 1d EMA34 as trend filter to avoid counter-trend trades.
# Designed for low trade frequency (20-50 trades/year) to minimize fee drag.
# Works in bull/bear markets by following 1d trend while using Camarilla breakouts for entries.

name = "4h_1d_Camarilla_R3_S3_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on previous day's range)
    camarilla_R3 = np.full(len(high_1d), np.nan)
    camarilla_S3 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Use previous day's high, low, close
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        range_val = ph - pl
        
        # Camarilla formulas
        camarilla_R3[i] = pc + range_val * 1.1 / 2
        camarilla_S3[i] = pc - range_val * 1.1 / 2
    
    # Trend filter: 1d EMA34
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema_34
    trend_down = close_1d < ema_34
    
    # Align Camarilla levels and trend to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or
            np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + uptrend
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                trend_up_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + downtrend
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  trend_down_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR trend turns down
            if (close[i] < S3_aligned[i]) or \
               not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR trend turns up
            if (close[i] > R3_aligned[i]) or \
               not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals