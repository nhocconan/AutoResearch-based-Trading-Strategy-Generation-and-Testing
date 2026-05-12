#!/usr/bin/env python3
# 1H_CAMARILLA_R1_S1_BREAKOUT_4H_EMA50_TREND_VOLUME
# Hypothesis: Camarilla R1/S1 levels on 4h chart represent strong breakout points with trend and volume confirmation.
# Price breaking above R1 with volume and 4h uptrend signals continuation long.
# Price breaking below S1 with volume and 4h downtrend signals continuation short.
# Uses 4h for signal direction, 1h only for entry timing to reduce false breakouts.
# Session filter (08-20 UTC) reduces noise trades. Target: 15-37 trades/year on 1h timeframe.

name = "1H_CAMARILLA_R1_S1_BREAKOUT_4H_EMA50_TREND_VOLUME"
timeframe = "1h"
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
    
    # 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R1 and S1 levels from previous 4h bar (requires previous bar's data)
    camarilla_r1 = np.full(len(close_4h), np.nan)
    camarilla_s1 = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        # Previous 4h bar's values
        ph = high_4h[i-1]
        pl = low_4h[i-1]
        pc = close_4h[i-1]
        range_val = ph - pl
        
        # Camarilla R1 and S1 levels
        camarilla_r1[i] = pc + range_val * 1.1 / 6
        camarilla_s1[i] = pc - range_val * 1.1 / 6
    
    # EMA50 for 4h trend filter
    ema50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current 1h volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    # Align all 4h data to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to ensure previous bar data exists
        # Skip if any critical data is not ready or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or not session_mask[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with volume spike in uptrend
            if (high[i] > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 with volume spike in downtrend
            elif (low[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R1 or trend reversal
            if (close[i] < camarilla_r1_aligned[i] or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price rises back above S1 or trend reversal
            if (close[i] > camarilla_s1_aligned[i] or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals