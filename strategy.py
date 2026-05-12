#!/usr/bin/env python3
"""
12H_CAMARILLA_R1_S1_BREAKOUT_1D_VOLUME_SPIKE
Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance.
Price breaking above R1 or below S1 with volume spike indicates institutional interest.
Volume confirmation reduces false breakouts. Designed for ~15-25 trades/year on 12h
to minimize fee decay while capturing meaningful market moves in both bull and bear markets.
"""
name = "12H_CAMARILLA_R1_S1_BREAKOUT_1D_VOLUME_SPIKE"
timeframe = "12h"
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
    
    # Volume spike: volume > 2.0 * 30-period average (strict filter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1D data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C,H,L are close, high, low of previous day
    prev_close = df_1d['close'].shift(1).values  # previous day close
    prev_high = df_1d['high'].shift(1).values    # previous day high
    prev_low = df_1d['low'].shift(1).values      # previous day low
    
    # Calculate R1 and S1 for each day
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align daily Camarilla levels to 12h timeframe (wait for day to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup for volume MA
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike
            if close[i] > camarilla_r1_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike
            elif close[i] < camarilla_s1_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns below R1 (false breakout) or reverses below midpoint
            # Midpoint between R1 and S1 for re-entry prevention
            midpoint = (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2
            if close[i] < camarilla_r1_aligned[i] or close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns above S1 or reverses above midpoint
            midpoint = (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2
            if close[i] > camarilla_s1_aligned[i] or close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals