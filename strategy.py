#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Filter
Hypothesis: Camarilla pivot points (R1/S1) act as strong support/resistance levels. 
Breakouts above R1 or below S1 with volume confirmation, filtered by 4h trend (EMA50), 
provide high-probability entries. Session filter (08-20 UTC) reduces noise. 
Target: 15-30 trades/year per symbol by using 4h trend for direction and 1h for timing.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Filter"
timeframe = "1h"
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
    
    # Get 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h trend filter: EMA(50) on close
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels using previous day's OHLC
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values  # Previous day close
    prev_high = df_1d['high'].shift(1).values    # Previous day high
    prev_low = df_1d['low'].shift(1).values      # Previous day low
    
    # Calculate Camarilla levels
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.1 / 12
    s1 = prev_close - rang * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        if position == 0:
            # LONG: Price breaks above R1, volume confirmation, 4h uptrend, session active
            if (close[i] > r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50_4h_aligned[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1, volume confirmation, 4h downtrend, session active
            elif (close[i] < s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR 4h trend turns down OR session ends
            if (close[i] < s1_aligned[i] or 
                close[i] < ema50_4h_aligned[i] or 
                not session_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR 4h trend turns up OR session ends
            if (close[i] > r1_aligned[i] or 
                close[i] > ema50_4h_aligned[i] or 
                not session_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals