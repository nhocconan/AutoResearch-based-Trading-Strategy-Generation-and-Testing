#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance. A breakout above R3 or below S3 with volume confirmation and aligned 1d trend (close > EMA34) signals continuation. Uses 0.30 position size to balance risk/return and limit trade frequency (~20-40/year) to minimize fee drag in 4-hour bars. Works in both bull and bear markets by following the dominant 1d trend.
"""

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for trend filter and Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend filter: EMA(34) on close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # Note: We use the previous day's HLC to avoid look-ahead
    camarilla_high = df_1d['high'].values
    camarilla_low = df_1d['low'].values
    camarilla_close = df_1d['close'].values
    
    camarilla_range = camarilla_high - camarilla_low
    r3 = camarilla_close + 1.1 * camarilla_range
    s3 = camarilla_close - 1.1 * camarilla_range
    
    # Align Camarilla levels to 4h timeframe (they update only when new 1d bar forms)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous bar for breakout
        if position == 0:
            # LONG: Close breaks above R3, volume confirmation, price above 1d EMA34 (uptrend)
            if (close[i] > r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Close breaks below S3, volume confirmation, price below 1d EMA34 (downtrend)
            elif (close[i] < s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close breaks below S3 (reversal) OR volume drops significantly
            if (close[i] < s3_aligned[i]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Close breaks above R3 (reversal) OR volume drops significantly
            if (close[i] > r3_aligned[i]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals