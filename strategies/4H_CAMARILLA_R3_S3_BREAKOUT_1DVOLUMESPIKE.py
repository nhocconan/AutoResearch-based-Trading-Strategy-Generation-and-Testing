#!/usr/bin/env python3
"""
4H_CAMARILLA_R3_S3_BREAKOUT_1DVOLUMESPIKE
Hypothesis: Camarilla pivot levels (R3/S3) derived from daily OHLC act as strong support/resistance.
A breakout above R3 or below S3 with volume confirmation (>1.5x 20-bar average) indicates momentum.
Use 1d EMA34 as trend filter to avoid counter-trend trades. Designed for ~25-35 trades/year on 4h
to minimize fee drag while capturing institutional breakout moves in both bull and bear markets.
"""
name = "4H_CAMARILLA_R3_S3_BREAKOUT_1DVOLUMESPIKE"
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
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 2
    S3 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after warmup for EMA34
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 + volume spike + above 1d EMA34 (uptrend)
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 + volume spike + below 1d EMA34 (downtrend)
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Camarilla interior (between S3 and R3) OR trend reversal
            if (close[i] < R3_aligned[i] and close[i] > S3_aligned[i]) or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Camarilla interior OR trend reversal
            if (close[i] < R3_aligned[i] and close[i] > S3_aligned[i]) or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals