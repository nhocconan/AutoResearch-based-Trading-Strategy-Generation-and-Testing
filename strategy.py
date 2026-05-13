#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and session filter (08-20 UTC).
# Enters long when price breaks above R1 level with 4h bullish trend (close > EMA20) during active session.
# Enters short when price breaks below S1 level with 4h bearish trend (close < EMA20) during active session.
# Exits when price reverts to the 4h EMA50 (mean reversion in range).
# Uses discrete position sizing (0.20) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~15-37/year) to work in both bull and bear markets by using 4h trend filter.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Session"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 (based on previous 1d bar)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 4h data for trend filter (EMA20)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 4h EMA50 for exit condition
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with 4h bullish trend
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 with 4h bearish trend
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema20_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to 4h EMA50 (mean reversion in range)
            if close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reverts to 4h EMA50 (mean reversion in range)
            if close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals