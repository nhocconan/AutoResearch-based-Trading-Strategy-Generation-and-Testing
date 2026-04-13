#!/usr/bin/env python3
"""
4h_1d_Engulfing_Engulfing_Signal_Strategy
Hypothesis: Bullish and bearish engulfing candlestick patterns on the 4h chart,
when aligned with the daily trend (EMA50) and confirmed by volume expansion,
provide high-probability entries for trend continuation.
The daily trend filter ensures we trade with the higher timeframe momentum,
while volume expansion confirms institutional participation.
This pattern-based approach works in both bull and bear markets by capturing
strong directional moves after consolidation.
Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close >= open_price) & (open_price <= close) & \
                     (close > open_price.shift(1)) & (open_price < close.shift(1))
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (close < open_price) & (open_price > close) & \
                     (close <= open_price) & (open_price >= close) & \
                     (close < open_price.shift(1)) & (open_price > close.shift(1))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(1, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Bullish engulfing pattern
        # 2. Price above daily EMA50 (1d trend filter)
        # 3. Volume expansion
        long_condition = bullish_engulf[i] and close[i] > ema_50_1d_aligned[i] and volume_expansion[i]
        
        # Short conditions:
        # 1. Bearish engulfing pattern
        # 2. Price below daily EMA50 (1d trend filter)
        # 3. Volume expansion
        short_condition = bearish_engulf[i] and close[i] < ema_50_1d_aligned[i] and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Engulfing_Engulfing_Signal_Strategy"
timeframe = "4h"
leverage = 1.0