#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS
Hypothesis: Camarilla pivot levels (R3/S3) from daily chart provide strong support/resistance levels.
Breakout of these levels with volume confirmation and daily trend filter captures institutional moves.
Works in bull markets by catching breakouts of resistance and in bear markets by catching breakdowns of support.
Uses 12h timeframe for lower frequency to reduce fee drag, with 1d trend filter for higher win rate.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        # Camarilla formulas
        camarilla_r3[i] = prev_close + range_ * 1.1 / 4
        camarilla_s3[i] = prev_close - range_ * 1.1 / 4
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = np.zeros_like(close_1d)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    
    # Align 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12-period volume average for volume spike filter
    vol_ma_12 = np.zeros_like(volume)
    if len(volume) >= 12:
        vol_ma_12[11] = np.mean(volume[:12])
        for i in range(12, len(volume)):
            vol_ma_12[i] = (volume[i] + vol_ma_12[i-1] * 11) / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_12[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 12-period average
        vol_spike = volume[i] > 1.5 * vol_ma_12[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike and daily uptrend (close > EMA34)
            if (close[i] > camarilla_r3_aligned[i] and vol_spike and 
                close_1d[i] > ema_34_1d_aligned[i] if not np.isnan(ema_34_1d_aligned[i]) else True):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and daily downtrend (close < EMA34)
            elif (close[i] < camarilla_s3_aligned[i] and vol_spike and 
                  close_1d[i] < ema_34_1d_aligned[i] if not np.isnan(ema_34_1d_aligned[i]) else True):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or loss of volume/spike
            if (close[i] < camarilla_s3_aligned[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or loss of volume/spike
            if (close[i] > camarilla_r3_aligned[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals