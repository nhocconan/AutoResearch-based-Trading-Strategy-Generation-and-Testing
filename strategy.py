#!/usr/bin/env python3
# 4h_Three_White_Soldiers_Black_Crows_1dTrend
# Strategy: Trade three consecutive bullish/bearish candles with 1d trend filter
# Long when three consecutive bullish candles close above 1d EMA(50)
# Short when three consecutive bearish candles close below 1d EMA(50)
# Exit when opposite pattern forms or trend weakens
# Uses price action patterns with trend filter to capture momentum in both bull and bear markets
# Designed for 4h timeframe with selective entries to minimize trade frequency

name = "4h_Three_White_Soldiers_Black_Crows_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Identify bullish and bearish candles
    bullish = close > open_price  # Close > Open
    bearish = close < open_price  # Close < Open
    
    # Count consecutive bullish/bearish candles
    consec_bullish = np.zeros(n, dtype=int)
    consec_bearish = np.zeros(n, dtype=int)
    
    for i in range(1, n):
        if bullish[i]:
            consec_bullish[i] = consec_bullish[i-1] + 1
            consec_bearish[i] = 0
        elif bearish[i]:
            consec_bearish[i] = consec_bearish[i-1] + 1
            consec_bullish[i] = 0
        else:
            consec_bullish[i] = 0
            consec_bearish[i] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Three consecutive bullish candles and above 1d EMA50 (uptrend filter)
            if consec_bullish[i] >= 3 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Three consecutive bearish candles and below 1d EMA50 (downtrend filter)
            elif consec_bearish[i] >= 3 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Three consecutive bearish candles or price below 1d EMA50
            if consec_bearish[i] >= 3 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Three consecutive bullish candles or price above 1d EMA50
            if consec_bullish[i] >= 3 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals