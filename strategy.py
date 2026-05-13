#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R3/S3) from daily chart act as strong support/resistance.
Breakout above R3 or below S3 with volume spike and aligned daily trend (EMA34) captures
institutional moves. Works in bull/bear by following breakouts with volume confirmation.
Target: 15-35 trades/year on 12h timeframe to minimize fee drag.
"""

name = "12h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day: R3, S3
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    hl_range = df_1d['high'] - df_1d['low']
    r3 = df_1d['close'] + 1.1 * hl_range / 2
    s3 = df_1d['close'] - 1.1 * hl_range / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_ltf_to_htf(prices, df_1d, r3.values)
    s3_aligned = align_ltf_to_htf(prices, df_1d, s3.values)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_aligned = align_ltf_to_htf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Break above R3 with volume spike and above daily EMA34
            if (close[i] > r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike and below daily EMA34
            elif (close[i] < s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < trend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R3 or below daily EMA34
            if (close[i] < r3_aligned[i] or 
                close[i] < trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S3 or above daily EMA34
            if (close[i] > s3_aligned[i] or 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals