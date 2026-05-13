#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels from 1-day provide strong support/resistance. 
Breakouts above R3 or below S3 with volume confirmation and aligned 1-day trend 
(close > EMA34 for long, close < EMA34 for short) signal continuation. 
Uses 25% position size targeting 20-50 trades/year to minimize fee drag in 6-hour bars.
Works in both bull and bear markets by following the 1-day trend direction.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Using previous day's OHLC for current levels (non-lookahead)
    day_open = df_1d['open'].values
    day_high = df_1d['high'].values
    day_low = df_1d['low'].values
    day_close = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 from previous day
    r3 = day_close + (day_high - day_low) * 1.1 / 2
    s3 = day_close - (day_high - day_low) * 1.1 / 2
    
    # Align to 6h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d trend filter: EMA(34) on close
    ema34_1d = pd.Series(day_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 24-period average (4 days worth of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after volume MA warmup
        if position == 0:
            # LONG: Break above R3, volume confirmation, price above 1d EMA34 (uptrend)
            if (close[i] > r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3, volume confirmation, price below 1d EMA34 (downtrend)
            elif (close[i] < s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below R3 (failed breakout) OR trend reversal
            if (close[i] < r3_aligned[i]) or \
               (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above S3 (failed breakdown) OR trend reversal
            if (close[i] > s3_aligned[i]) or \
               (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals