#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from daily chart act as key support/resistance. 
Buy when price breaks above R1 with volume confirmation and weekly uptrend; 
Sell when price breaks below S1 with volume confirmation and weekly downtrend.
Weekly trend filter reduces whipsaws in sideways markets. Works in both bull and bear regimes by following higher timeframe direction.
Target: 12-37 trades/year per symbol.
"""

name = "12h_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
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
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # Since we don't have direct access to previous day, we'll use rolling window
    # This approximates the previous day's levels for intraday calculation
    lookback = 24  # 24 * 12h = 12 days worth of data to approximate daily
    if n < lookback:
        return np.zeros(n)
    
    # Rolling max/min/close for previous day approximation
    # We use shift(1) to avoid look-ahead: use data from previous bar only
    prev_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prev_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    prev_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Calculate pivot and Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1 = prev_close + (range_val * 1.1 / 12)
    s1 = prev_close - (range_val * 1.1 / 12)
    
    # Volume confirmation: current volume > 1.5 * average volume
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 * 12h = 12 days
    volume_ok = volume > (vol_ma * 1.5)
    
    # Weekly trend filter: EMA50 on weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: price breaks above R1, volume confirmation, weekly uptrend
            if close[i] > r1[i] and volume_ok[i] and uptrend_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1, volume confirmation, weekly downtrend
            elif close[i] < s1[i] and volume_ok[i] and downtrend_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 (reversal) or volume drops
            if close[i] < s1[i] or not volume_ok[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 (reversal) or volume drops
            if close[i] > r1[i] or not volume_ok[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals