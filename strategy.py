#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance. Breakouts above R3 or below S3 with volume spikes and weekly trend filter capture institutional moves. Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drift. Works in both bull and bear regimes by following confirmed weekly trends with volume confirmation.
"""

name = "4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for Camarilla pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each day
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 4
    camarilla_s3 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 4
    
    # Align daily pivot levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Break above R3 with volume spike and above weekly EMA50
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike and below weekly EMA50
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < trend_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price drops below S3 or weekly trend turns down
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < trend_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R3 or weekly trend turns up
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals