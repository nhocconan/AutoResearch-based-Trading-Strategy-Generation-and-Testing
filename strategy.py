#!/usr/bin/env python3
"""
4h_Engulfing_Pattern_Trend_Follow
Hypothesis: Bullish/bearish engulfing candles indicate momentum shifts. Combined with 
1-day EMA50 trend filter and volume confirmation, this captures trend continuations 
while avoiding counter-trend trades. Designed for low trade frequency (20-40/year) 
to work in both bull and bear markets by following established trends with 
momentum confirmation.
"""

name = "4h_Engulfing_Pattern_Trend_Follow"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    # Detect bullish and bearish engulfing patterns
    bullish_engulf = (close > open_price) & (open_price > close) & (close >= open_price) & (open_price <= close)
    bearish_engulf = (close < open_price) & (open_price < close) & (close <= open_price) & (open_price >= close)
    # Actually: bullish engulf = current bullish candle engulfs previous bearish candle
    bullish_engulf = (close > open_price) & (close >= open_price) & (open_price <= close) & (open_price > close)
    # Fix: proper engulfing detection
    bullish_engulf = (close > open_price) & (open_price < close) & (close >= open_price) & (open_price <= close)
    # Correct implementation:
    bullish_engulf = (close > open_price) & (open_price < close) & (close >= open_price[1:]) & (open_price <= close[1:])  # This approach needs fixing
    
    # Proper engulfing pattern detection
    bullish_engulf = np.zeros(n, dtype=bool)
    bearish_engulf = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        # Bullish engulf: current green candle completely engulfs previous red candle
        if close[i] > open_price[i] and close[i-1] < open_price[i-1]:
            if close[i] >= open_price[i-1] and open_price[i] <= close[i-1]:
                bullish_engulf[i] = True
        # Bearish engulf: current red candle completely engulfs previous green candle
        if close[i] < open_price[i] and close[i-1] > open_price[i-1]:
            if close[i] <= open_price[i-1] and open_price[i] >= close[i-1]:
                bearish_engulf[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Bullish engulf + above daily EMA50 + volume confirmation
            if bullish_engulf[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish engulf + below daily EMA50 + volume confirmation
            elif bearish_engulf[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish engulf or price drops below EMA50
            if bearish_engulf[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish engulf or price rises above EMA50
            if bullish_engulf[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals