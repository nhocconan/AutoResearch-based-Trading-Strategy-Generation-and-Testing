#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend
Hypothesis: 1-hour breakouts above R1 or below S1 of the 4-hour Camarilla pivot levels
with volume confirmation and aligned 4-hour trend (close > EMA21) capture short-term
momentum in both bull and bear markets. Uses 4-hour trend for directional bias and
1-hour for precise entry timing, targeting 15-35 trades/year to minimize fee drag.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla pivots and trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla pivot levels for 4h
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # Where C, H, L are from previous 4h bar
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Calculate pivot levels
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h trend filter: EMA(21) on close
    ema21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Volume confirmation: current volume > 1.5x 12-period average (6 hours on 1h)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(12, n):  # Start after warmup
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Breakout above R1 with volume confirmation and uptrend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema21_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Breakdown below S1 with volume confirmation and downtrend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema21_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend reverses
            if (close[i] < camarilla_r1_aligned[i]) or \
               (close[i] < ema21_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend reverses
            if (close[i] > camarilla_s1_aligned[i]) or \
               (close[i] > ema21_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals