#!/usr/bin/env python3
# Hypothesis: 6h strategy using 1d Williams %R (mean reversion) with 12h EMA20 trend filter.
# Long when Williams %R < -80 (oversold) and close > 12h EMA20 (uptrend).
# Short when Williams %R > -20 (overbought) and close < 12h EMA20 (downtrend).
# Williams %R calculated on 1d high/low/close, aligned to 6h with no extra delay (EWM-like).
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Williams %R is a bounded oscillator that identifies exhaustion points; EMA20 filter ensures
# trades align with intermediate trend, reducing whipsaws in ranging markets.
# Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets.

name = "6h_WilliamsR_MeanReversion_12hEMA20_Trend"
timeframe = "6h"
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
    
    # Calculate 12h EMA20 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after sufficient data for Williams %R
        # Skip if any required data is NaN
        if (np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Oversold (%R < -80) and uptrend (close > 12h EMA20)
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_20_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought (%R > -20) and downtrend (close < 12h EMA20)
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_20_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Either overbought (%R > -50) or trend change (close < 12h EMA20)
            if (williams_r_aligned[i] > -50 or 
                close[i] < ema_20_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Either oversold (%R < -50) or trend change (close > 12h EMA20)
            if (williams_r_aligned[i] < -50 or 
                close[i] > ema_20_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals