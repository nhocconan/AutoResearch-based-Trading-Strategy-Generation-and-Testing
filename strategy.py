#!/usr/bin/env python3
# 4h_12h_Camarilla_R3_S3_Breakout_Trend_Filter
# Hypothesis: Combines 12h Camarilla pivot levels (R3/S3) with 4h breakouts for trend-following entries.
# Uses 12h trend filter (EMA50 slope) to avoid counter-trend trades, and volume confirmation (>1.5x 20-period average)
# to filter for institutional participation. Designed for low trade frequency (<200 total 4h trades) to minimize fee drag.
# Works in bull/bear markets by following 12h trend while using 4h breaks of Camarilla R3/S3 levels for precise entries.

name = "4h_12h_Camarilla_R3_S3_Breakout_Trend_Filter"
timeframe = "4h"
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
    
    # Volume spike: >1.5x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 12h data for Camarilla pivot levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h timeframe
    # Typical Price = (H + L + C) / 3
    typical_price = (high_12h + low_12h + close_12h) / 3
    # Camarilla levels: R3 = TP + (H-L) * 1.1/2, S3 = TP - (H-L) * 1.1/2
    camarilla_r3 = typical_price + (high_12h - low_12h) * 1.1 / 2
    camarilla_s3 = typical_price - (high_12h - low_12h) * 1.1 / 2
    
    # 12h trend filter: EMA50 slope (positive = uptrend, negative = downtrend)
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = np.diff(ema_50, prepend=ema_50[0])
    
    # Align 12h indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_50_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend (EMA50 slope > 0) + price breaks above Camarilla R3 + volume spike
            if (ema_50_slope_aligned[i] > 0 and 
                close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (EMA50 slope < 0) + price breaks below Camarilla S3 + volume spike
            elif (ema_50_slope_aligned[i] < 0 and 
                  close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR trend turns down
            if (close[i] < camarilla_s3_aligned[i]) or \
               (ema_50_slope_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR trend turns up
            if (close[i] > camarilla_r3_aligned[i]) or \
               (ema_50_slope_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals