#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeS_v3
Hypothesis: Using tighter volume confirmation (2x average) and stricter momentum filter (close in upper/lower quarter) reduces overtrading while maintaining edge. The combination of Camarilla R1/S1 breakouts with 12h EMA50 trend filter works across regimes by only taking trades aligned with higher timeframe momentum, reducing false breakouts. Target: 20-40 trades/year per symbol.
"""

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeS_v3"
timeframe = "4h"
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
    
    # Volume spike: >2x 30-period average (stricter than before)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Daily range for momentum filter: close in upper/lower quarter (stricter)
    daily_range = high_1d - low_1d
    upper_quarter = low_1d + (daily_range * 3/4)
    lower_quarter = low_1d + (daily_range * 1/4)
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    upper_quarter_aligned = align_htf_to_ltf(prices, df_1d, upper_quarter)
    lower_quarter_aligned = align_htf_to_ltf(prices, df_1d, lower_quarter)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(upper_quarter_aligned[i]) or
            np.isnan(lower_quarter_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + 12h EMA50 uptrend + volume spike + close in upper quarter of daily range
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i] and
                close[i] > upper_quarter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 12h EMA50 downtrend + volume spike + close in lower quarter of daily range
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i] and
                  close[i] < lower_quarter_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR closes below 12h EMA50
            if (close[i] < camarilla_s1_aligned[i]) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR closes above 12h EMA50
            if (close[i] > camarilla_r1_aligned[i]) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals