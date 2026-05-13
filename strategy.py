#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above R1 with volume > 1.5x MA20 and close > 12h EMA50.
# Short when price breaks below S1 with volume > 1.5x MA20 and close < 12h EMA50.
# Exit on opposite Camarilla level touch (S1 for long, R1 for short) or trend failure.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 25-40 trades/year.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume_v1"
timeframe = "4h"
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for current day using previous day's OHLC
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    r1 = close_1d + camarilla_range
    s1 = close_1d - camarilla_range
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels for current day trading)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1, additional_delay_bars=1)  # previous day's R1
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1, additional_delay_bars=1)  # previous day's S1
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R1 with volume confirmation and 12h EMA50 uptrend
            if close[i] > r1_aligned[i] and volume_confirm[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25  # 25% position
                position = 1
            # SHORT: price breaks below S1 with volume confirmation and 12h EMA50 downtrend
            elif close[i] < s1_aligned[i] and volume_confirm[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25  # 25% position
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches S1 (opposite level) or trend fails (price < 12h EMA50)
            if close[i] < s1_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches R1 (opposite level) or trend fails (price > 12h EMA50)
            if close[i] > r1_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals