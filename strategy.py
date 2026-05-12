#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Price breaking above/below Camarilla R3/S3 levels (derived from 1w high-low-close) with 1w EMA trend filter and volume confirmation (1.5x average) captures strong trending moves while avoiding false breakouts. R3/S3 levels represent stronger support/resistance than R1/S1, reducing false signals. Works in bull/bear by following 1w trend direction. Target: 30-100 trades over 4 years (7-25/year).
"""

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from 1w data
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # where C = close, H = high, L = low of previous week
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Shift by 1 to use previous week's data
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    camarilla_upper = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 4
    camarilla_lower = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_1w, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_1w, camarilla_lower)
    
    # 1w EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: >1.5x 20-period average (1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 + 1w EMA34 uptrend + volume spike
            if (close[i] > camarilla_upper_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + 1w EMA34 downtrend + volume spike
            elif (close[i] < camarilla_lower_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S3 (reversal level)
            if close[i] < camarilla_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R3 (reversal level)
            if close[i] > camarilla_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals