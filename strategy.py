#!/usr/bin/env python3
"""
12h_Weekly_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R3, S3) from weekly timeframe breakouts on 12h timeframe,
confirmed by weekly trend (price > weekly EMA34) and volume spikes (>2x 24-period average),
capture momentum continuation in both bull and bear markets. Weekly pivot levels act as
strong support/resistance, and breakouts with volume confirmation indicate institutional interest.
Position size 0.25 limits risk; exit when price re-enters the weekly Camarilla levels
or trend reverses. Targets 15-30 trades/year to minimize fee drift.
"""

name = "12h_Weekly_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
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
    
    # Get weekly data for Camarilla pivots and trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly trend filter: EMA(34) on close
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate weekly Camarilla levels: R3, S3
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use R3 and S3 as breakout levels
    H = df_1w['high'].values
    L = df_1w['low'].values
    C = df_1w['close'].values
    
    # Calculate Camarilla levels for each weekly bar
    camarilla_R3 = C + ((H - L) * 1.1 / 4)
    camarilla_S3 = C - ((H - L) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (weekly levels are constant within the week)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S3)
    
    # Volume confirmation: current volume > 2.0x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after warmup for EMA34
        if position == 0:
            # LONG: Breakout above weekly R3 with volume confirmation and uptrend
            if (close[i] > camarilla_R3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly S3 with volume confirmation and downtrend
            elif (close[i] < camarilla_S3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below weekly R3 or trend reverses
            if (close[i] < camarilla_R3_aligned[i]) or \
               (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above weekly S3 or trend reverses
            if (close[i] > camarilla_S3_aligned[i]) or \
               (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals