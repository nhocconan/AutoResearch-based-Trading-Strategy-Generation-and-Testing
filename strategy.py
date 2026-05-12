#!/usr/bin/env python3
# 1d_1w_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Uses weekly Camarilla pivot levels (R3/S3) for breakout entries on daily timeframe.
# Trend filtered by weekly EMA50 to ensure alignment with higher timeframe direction.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Designed for low trade frequency (<150 total daily trades) to minimize fee drift.
# Works in bull/bear markets by following weekly trend direction while using weekly pivot levels for precise entries.

name = "1d_1w_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Volume spike: >1.5x 20-period average (on daily timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly data for Camarilla pivot levels and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for Camarilla calculation
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close[0] = close_1w[0]  # Handle first value
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    
    # Camarilla formulas
    range_1w = prev_high - prev_low
    camarilla_r3 = prev_close + (range_1w * 1.1 / 4)  # R3 level
    camarilla_s3 = prev_close - (range_1w * 1.1 / 4)  # S3 level
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 + volume spike + price above weekly EMA50
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + volume spike + price below weekly EMA50
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Camarilla R3 OR closes below weekly EMA50
            if (close[i] < camarilla_r3_aligned[i]) or \
               close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Camarilla S3 OR closes above weekly EMA50
            if (close[i] > camarilla_s3_aligned[i]) or \
               close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals