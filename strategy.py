#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) on 12h timeframe act as key support/resistance.
Breakouts above R1 or below S1 with volume confirmation and 1d trend alignment capture
momentum moves while avoiding false breakouts in ranging markets. Exit on reversion to
the pivot point (PP) or trend reversal. Position size 0.25 limits risk and targets
~15-25 trades/year to minimize fee drag in both bull and bear markets.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels for each 12h bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, PP = (H+L+C)/3
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    camarilla_pp = (h_12h + l_12h + c_12h) / 3.0
    camarilla_r1 = c_12h + (h_12h - l_12h) * 1.1 / 12.0
    camarilla_s1 = c_12h - (h_12h - l_12h) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h chart (no additional delay needed)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above R1 with volume confirmation and uptrend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 with volume confirmation and downtrend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point or trend reverses
            if (close[i] < camarilla_pp_aligned[i]) or \
               (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point or trend reverses
            if (close[i] > camarilla_pp_aligned[i]) or \
               (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals