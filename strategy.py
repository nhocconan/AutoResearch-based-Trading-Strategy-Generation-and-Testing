#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from 1-day provide strong support/resistance. 
Breakouts above R1 or below S1 with volume confirmation and 1-day EMA50 trend filter 
capture institutional moves. Designed for low trade frequency (20-40/year) with 
clear entry/exit rules that work in both bull and bear markets by filtering 
counter-trend trades.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Calculate 1-day Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_s1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # 1-day trend filter: EMA 50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and uptrend filter
            if close[i] > camarilla_r1_aligned[i] and volume_confirm[i]:
                if close[i] > ema_50_1d_aligned[i]:  # Only long in uptrend
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S1 with volume confirmation and downtrend filter
            elif close[i] < camarilla_s1_aligned[i] and volume_confirm[i]:
                if close[i] < ema_50_1d_aligned[i]:  # Only short in downtrend
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (reversion to mean) or trend change
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (reversion to mean) or trend change
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals