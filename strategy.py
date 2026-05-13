#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend
Hypothesis: Camarilla pivot levels from weekly timeframe act as strong support/resistance.
Breakouts above R3 or below S3 with volume confirmation and aligned weekly trend signal strong momentum.
12-hour bars reduce trade frequency to target 12-37 trades/year. Weekly trend filter helps avoid whipsaws
in choppy markets and captures sustained moves in both bull and bear regimes. Position size 0.25 balances
risk and return while minimizing fee drag.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend"
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
    
    # Calculate weekly Camarilla pivot levels
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # Where C, H, L are from previous week
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Calculate pivot levels
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Weekly trend filter: EMA(34) on close
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 2.0x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above R3 with volume confirmation and uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 with volume confirmation and downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend reverses
            if (close[i] < camarilla_r3_aligned[i]) or \
               (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend reverses
            if (close[i] > camarilla_s3_aligned[i]) or \
               (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals