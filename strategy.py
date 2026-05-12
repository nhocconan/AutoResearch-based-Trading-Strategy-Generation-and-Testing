#!/usr/bin/env python3
# 12h_1d_Camarilla_R3_S3_Breakout_Trend_VolumeS
# Hypothesis: Uses 1d Camarilla R3 and S3 levels for breakout entries in the direction of the 1d trend.
# Trend determined by EMA34 on 1d timeframe. Volume confirmation (>1.5x 20-period average on 12h) filters for institutional participation.
# Designed for low trade frequency (~50-150 total over 4 years) to minimize fee drag on 12h timeframe.
# Works in bull/bear markets by following 1d trend while using Camarilla breakouts for precise entries.

name = "12h_1d_Camarilla_R3_S3_Breakout_Trend_VolumeS"
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
    
    # Volume spike: >1.5x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using previous day's OHLC)
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to calculate today's levels
        hl_range = high_1d[i-1] - low_1d[i-1]
        camarilla_r3[i] = close_1d[i-1] + 1.1 * hl_range
        camarilla_s3[i] = close_1d[i-1] - 1.1 * hl_range
    
    # First day has no previous day, so set to NaN
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # Trend: EMA34 on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend on 1d + price breaks above Camarilla R3 + volume spike
            if (uptrend_aligned[i] and 
                close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend on 1d + price breaks below Camarilla S3 + volume spike
            elif (downtrend_aligned[i] and 
                  close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR 1d trend turns down
            if (close[i] < camarilla_s3_aligned[i]) or \
               downtrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR 1d trend turns up
            if (close[i] > camarilla_r3_aligned[i]) or \
               uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals