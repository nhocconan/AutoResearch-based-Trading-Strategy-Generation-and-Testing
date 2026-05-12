#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_1W_TREND_FILTER
# Hypothesis: Camarilla R1/S1 levels from 1d provide strong support/resistance. 1w trend filter (EMA34) ensures we trade only in the direction of the weekly trend, avoiding counter-trend trades. Breakouts above R1 in uptrend or below S1 in downtrend capture momentum. Works in both bull and bear markets: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum within trend. Volume confirmation reduces false breakouts. Target: 20-40 trades/year on 4h timeframe.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_1W_TREND_FILTER"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # EMA34 for weekly trend
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1 (using previous day's OHLC)
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1w uptrend + price breaks above R1 + volume confirmation
            if (close[i] > ema34_1w_aligned[i] and 
                close[i] > camarilla_r1_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1w downtrend + price breaks below S1 + volume confirmation
            elif (close[i] < ema34_1w_aligned[i] and 
                  close[i] < camarilla_s1_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price breaks below S1 (reversal signal)
            if (close[i] <= ema34_1w_aligned[i] or 
                close[i] < camarilla_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price breaks above R1 (reversal signal)
            if (close[i] >= ema34_1w_aligned[i] or 
                close[i] > camarilla_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals