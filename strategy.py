#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R3_S3_Breakout_1dEMA34_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts with 1-day EMA34 trend filter and volume spikes capture institutional breakout patterns in both bull and bear markets. The Camarilla levels act as key pivot points where price often reverses or breaks out with volume. The EMA34 filter ensures alignment with the daily trend, while volume confirmation filters false breakouts. Designed for low trade frequency (20-40/year) with clear entry/exit rules.
"""

name = "4h_Camarilla_Pivot_R3_S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Calculate Camarilla levels from previous day
    # Using previous day's high, low, close
    prev_high = np.roll(high, 1)  # Previous bar's high
    prev_low = np.roll(low, 1)    # Previous bar's low  
    prev_close = np.roll(close, 1) # Previous bar's close
    
    # For first bar, use current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + (range_val * 1.1 / 2.0)  # R3 = pivot + 1.1*(H-L)/2
    S3 = pivot - (range_val * 1.1 / 2.0)  # S3 = pivot - 1.1*(H-L)/2
    
    # Calculate EMA 34 on 1-day timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 since we use previous bar data
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and above daily EMA34 (uptrend)
            if close[i] > R3[i] and volume_confirm[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume confirmation and below daily EMA34 (downtrend)
            elif close[i] < S3[i] and volume_confirm[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 (reversal level) or volume dries up
            if close[i] < S3[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (reversal level) or volume dries up
            if close[i] > R3[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals