#!/usr/bin/env python3
"""
1h 4h-1d Trend Alignment with Volume and Session Filter
Hypothesis: In trending markets, when 4h and 1d EMAs align (both bullish or bearish),
price pulls back to the 4h EMA during the active session (08-20 UTC) offers high-probability
entries with volume confirmation. This captures trend continuation moves while avoiding
counter-trend noise. Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d EMA21 once before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_4h = ema_21_4h_aligned[i]
        ema_1d = ema_21_1d_aligned[i]
        vol_conf = vol_ratio[i] > 1.3
        in_session = (8 <= hours[i] <= 20)
        
        if position == 0:
            # Long setup: both EMAs bullish, price near 4h EMA, volume, session
            if ema_4h > ema_1d and price > ema_4h * 0.998 and price < ema_4h * 1.002 and vol_conf and in_session:
                signals[i] = 0.20
                position = 1
            # Short setup: both EMAs bearish, price near 4h EMA, volume, session
            elif ema_4h < ema_1d and price < ema_4h * 1.002 and price > ema_4h * 0.998 and vol_conf and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit if trend alignment breaks or price moves significantly away from 4h EMA
            if ema_4h < ema_1d or price < ema_4h * 0.985 or price > ema_4h * 1.015:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit if trend alignment breaks or price moves significantly away from 4h EMA
            if ema_4h > ema_1d or price > ema_4h * 1.015 or price < ema_4h * 0.985:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_EMA21_Align_Volume_Session"
timeframe = "1h"
leverage = 1.0