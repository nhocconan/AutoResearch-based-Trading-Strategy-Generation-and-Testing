#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla R3/S3 breakouts on 4h with 1d EMA trend filter and volume confirmation capture strong momentum moves in both bull and bear markets. Uses tight entry (Camarilla R3/S3) to limit trades (~25-40/year) and avoid fee drag. Exit on reversion to Camarilla R4/S4 or trend reversal.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Calculate Camarilla levels from previous 4h bar
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # Shift by 1 to use previous bar's levels (no look-ahead)
    shift_high = np.roll(high, 1)
    shift_low = np.roll(low, 1)
    shift_close = np.roll(close, 1)
    shift_high[0] = high[0]
    shift_low[0] = low[0]
    shift_close[0] = close[0]
    
    camarilla_r3 = shift_close + (shift_high - shift_low) * 1.1 / 4.0
    camarilla_s3 = shift_close - (shift_high - shift_low) * 1.1 / 4.0
    camarilla_r4 = shift_close + (shift_high - shift_low) * 1.1 / 2.0
    camarilla_s4 = shift_close - (shift_high - shift_low) * 1.1 / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above Camarilla R3 with volume confirmation and uptrend
            if (close[i] > camarilla_r3[i] and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Camarilla S3 with volume confirmation and downtrend
            elif (close[i] < camarilla_s3[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla R4 or trend reverses
            if (close[i] >= camarilla_r4[i]) or \
               (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla S4 or trend reverses
            if (close[i] <= camarilla_s4[i]) or \
               (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals