# 12h_1d_Camarilla_R3_S3_Breakout_Volume
# Hypothesis: Camarilla R3/S3 levels from daily timeframe act as strong support/resistance on 12h chart.
# Breakouts above R3 or below S3 with volume confirmation capture momentum moves in both bull and bear markets.
# Position size 0.25 targets ~20-40 trades/year to minimize fee drag.
# Uses 1d HTF for Camarilla calculation, 12h for execution.

name = "12h_1d_Camarilla_R3_S3_Breakout_Volume"
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
    
    # Calculate daily Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 2.0
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 2.0
    
    # Align daily Camarilla levels to 12h chart (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above daily R3 with volume confirmation
            if close[i] > camarilla_r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below daily S3 with volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily pivot point or breaks S3
            daily_pp = (h_1d[i] + l_1d[i] + c_1d[i]) / 3.0
            daily_pp_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(h_1d, daily_pp))[i]
            if close[i] < camarilla_s3_aligned[i] or close[i] < daily_pp_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to daily pivot point or breaks R3
            daily_pp = (h_1d[i] + l_1d[i] + c_1d[i]) / 3.0
            daily_pp_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(h_1d, daily_pp))[i]
            if close[i] > camarilla_r3_aligned[i] or close[i] > daily_pp_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals