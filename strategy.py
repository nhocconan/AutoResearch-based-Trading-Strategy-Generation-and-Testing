#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe provide high-probability
breakout/breakdown levels. Combined with weekly trend filter (price > weekly EMA50) and
volume confirmation (current volume > 1.5x 24-period average), this captures strong
momentum moves. Designed for 12h timeframe to limit trades (~15-30/year) and reduce fee
drag. Works in bull markets via upside breaks and in bear markets via downside breaks.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend"
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
    
    # Get daily data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d (R1, S1)
    # Using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly trend filter: EMA(50) on weekly close
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above R1 with volume confirmation and weekly uptrend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 with volume confirmation and weekly downtrend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or weekly trend reverses
            if (close[i] < camarilla_r1_aligned[i]) or \
               (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or weekly trend reverses
            if (close[i] > camarilla_s1_aligned[i]) or \
               (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals