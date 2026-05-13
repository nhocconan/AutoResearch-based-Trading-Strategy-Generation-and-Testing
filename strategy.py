#/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla R3/S3 levels derived from daily high/low/close act as strong support/resistance.
Breakouts above daily R3 or below S3 with volume confirmation and daily trend alignment capture momentum moves.
Exit on reversion to daily pivot point (PP) or trend reversal. Position size 0.25 targets ~20-30 trades/year.
Works in both bull (breakouts with trend) and bear (mean reversion at extremes) markets via trend filter.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla pivot: PP = (H+L+C)/3
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    daily_pp = (h_1d + l_1d + c_1d) / 3.0
    daily_r3 = c_1d + (h_1d - l_1d) * 1.1 / 2.0
    daily_s3 = c_1d - (h_1d - l_1d) * 1.1 / 2.0
    
    # Align daily levels to 12h chart (wait for daily close)
    daily_pp_aligned = align_htf_to_ltf(prices, df_1d, daily_pp)
    daily_r3_aligned = align_htf_to_ltf(prices, df_1d, daily_r3)
    daily_s3_aligned = align_htf_to_ltf(prices, df_1d, daily_s3)
    
    # Daily trend filter: EMA50
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above daily R3 with volume confirmation and uptrend
            if (close[i] > daily_r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below daily S3 with volume confirmation and downtrend
            elif (close[i] < daily_s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily pivot or trend reverses
            if (close[i] < daily_pp_aligned[i]) or \
               (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to daily pivot or trend reverses
            if (close[i] > daily_pp_aligned[i]) or \
               (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals