# 12h_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Daily Camarilla R3/S3 levels act as strong support/resistance on 12h chart.
# Breakouts above R3 or below S3 with volume confirmation and daily EMA trend filter capture momentum.
# Uses 12h for execution and 1d EMA for trend direction. Target ~20-50 trades/year to avoid fee drag.
# Works in bull (breakouts with trend) and bear (breakdowns against trend filtered by 1d EMA).

name = "12h_Camarilla_R3_S3_Breakout_Trend_Volume"
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
    
    # Get daily EMA for trend filter
    ema_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above daily R3 with volume confirmation and daily EMA uptrend
            if close[i] > camarilla_r3_aligned[i] and volume_filter[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below daily S3 with volume confirmation and daily EMA downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume_filter[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily S3 or breaks below daily EMA
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to daily R3 or breaks above daily EMA
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals