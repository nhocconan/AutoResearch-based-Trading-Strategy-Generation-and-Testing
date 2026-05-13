#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R3/S3) derived from 1d high/low/close act as key intraday support/resistance. 
Breakouts above R3 or below S3 on 12h timeframe, confirmed by 1d uptrend/downtrend (price > EMA50/< EMA50) and volume spikes (>2x 24-period average), capture momentum continuation. 
Exit when price re-enters between R3 and S3 or trend reverses. Targets 12-37 trades/year to minimize fee drift.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla levels and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels: R3, S3 from 1d OHLC
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    hl_range = df_1d['high'] - df_1d['low']
    r3 = df_1d['close'] + hl_range * 1.1 / 4
    s3 = df_1d['close'] - hl_range * 1.1 / 4
    r3_vals = r3.values
    s3_vals = s3.values
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    
    # 1d trend filter: EMA(50) on close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after volume MA warmup
        if position == 0:
            # LONG: Breakout above R3 with volume confirmation and uptrend
            if (close[i] > r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 with volume confirmation and downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend reverses
            if (close[i] < r3_aligned[i]) or \
               (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend reverses
            if (close[i] > s3_aligned[i]) or \
               (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals