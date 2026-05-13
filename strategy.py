#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend
Hypothesis: Camarilla R3/S3 levels on 1d timeframe act as key support/resistance for 12h price action. 
Breakouts above R3 or below S3 with volume confirmation and aligned 1d trend (close > EMA34) signal strong momentum continuation. 
Uses 0.25 position size to balance risk/return and target 12-25 trades/year on 12h timeframe.
Works in bull markets via breakout continuation and in bear markets via breakdown continuation.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # Where C, H, L are from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d trend filter: EMA(34) on close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average (10 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above R3 with volume confirmation and uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 with volume confirmation and downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend reverses
            if (close[i] < camarilla_r3_aligned[i]) or \
               (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend reverses
            if (close[i] > camarilla_s3_aligned[i]) or \
               (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals