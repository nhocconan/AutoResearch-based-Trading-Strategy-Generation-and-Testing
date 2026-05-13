#!/usr/bin/env python3
"""
12h_Weekly_Camarilla_R3_S3_Breakout_WeeklyTrend
Hypothesis: Weekly timeframe acts as a strong trend filter. Breakouts above R3 or below S3 
on daily timeframe with volume confirmation and aligned weekly trend (close > EMA21 weekly) 
signal strong momentum continuation. Uses 0.25 position size to balance risk/return and 
limit trade frequency (~15-30/year) to minimize fee drag in 12-hour bars.
Works in bull markets via breakout continuation and in bear markets via breakdown continuation.
"""

name = "12h_Weekly_Camarilla_R3_S3_Breakout_WeeklyTrend"
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
    
    # Get daily and weekly data (once before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily Camarilla pivot levels (R3, S3)
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    # Where C, H, L are from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r3_1d = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    camarilla_s3_1d = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Weekly trend filter: EMA(21) on weekly close
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 1.5x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above R3 with volume confirmation and weekly uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 with volume confirmation and weekly downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or weekly trend reverses
            if (close[i] < camarilla_r3_aligned[i]) or \
               (close[i] < ema21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or weekly trend reverses
            if (close[i] > camarilla_s3_aligned[i]) or \
               (close[i] > ema21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals