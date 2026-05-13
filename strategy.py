#!/usr/bin/env python3
name = "6h_Chaikin_Oscillator_Zero_Cross_1D_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mpf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Chaikin Oscillator: (3-day EMA of ADL) - (10-day EMA of ADL)
    # ADL = ADL_prev + ((Close - Low) - (High - Close)) / (High - Low) * Volume
    adl = np.zeros(n)
    adl[0] = ((close[0] - low[0]) - (high[0] - close[0])) / (high[0] - low[0]) * volume[0] if high[0] != low[0] else 0
    for i in range(1, n):
        if high[i] != low[i]:
            money_flow_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
        else:
            money_flow_multiplier = 0
        adl[i] = adl[i-1] + money_flow_multiplier * volume[i]
    
    adl_series = pd.Series(adl)
    ema3_adl = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_adl = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3_adl - ema10_adl
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        if np.isnan(chaikin[i]) or np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Chaikin crosses above zero with daily uptrend
            if chaikin[i] > 0 and chaikin[i-1] <= 0 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin crosses below zero with daily downtrend
            elif chaikin[i] < 0 and chaikin[i-1] >= 0 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chaikin crosses below zero or trend reversal
            if chaikin[i] < 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin crosses above zero or trend reversal
            if chaikin[i] > 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals