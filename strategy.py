#!/usr/bin/env python3
"""
12h_1d_Camarilla_R3_S3_Breakout_Volume_v2
Hypothesis: Uses daily Camarilla R3/S3 levels as dynamic support/resistance on 12h chart.
Breakouts above R3 or below S3 with volume > 2x 20-period average and price above/below 50-period EMA capture momentum.
Includes volatility filter: only trade when ATR(14) > 0.5 * ATR(50) to avoid choppy markets.
Designed for low trade frequency (~20-40/year) to minimize fee drag in both bull and bear markets.
"""

name = "12h_1d_Camarilla_R3_S3_Breakout_Volume_v2"
timeframe = "12h"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 2.0
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 2.0
    
    # Align daily Camarilla levels to 12h chart (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate daily pivot point for exit
    camarilla_pivot = (h_1d + l_1d + c_1d) / 3.0
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: current volume > 2x 20-period average (stricter)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # EMA filter: price must be above/below 50-period EMA on 12h
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility filter: only trade when ATR(14) > 0.5 * ATR(50) to avoid chop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_14 > (0.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup for 50 EMA
        if position == 0:
            # LONG: Breakout above daily R3 with volume confirmation, price above EMA50, and sufficient volatility
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema_50[i] and 
                volatility_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below daily S3 with volume confirmation, price below EMA50, and sufficient volatility
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema_50[i] and 
                  volatility_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily pivot point or breaks S3
            if close[i] < camarilla_pivot_aligned[i] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to daily pivot point or breaks R3
            if close[i] > camarilla_pivot_aligned[i] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals