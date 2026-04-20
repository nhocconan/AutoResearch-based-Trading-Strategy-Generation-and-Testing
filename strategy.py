#!/usr/bin/env python3
# 6h_12h_ADX_Trend_With_Pullback_EMA
# Hypothesis: On 6h timeframe, use 12h ADX to identify strong trends (ADX > 25) and enter on pullbacks to 6h EMA21 in the direction of the trend.
# Long: 12h ADX > 25, 6h close > 6h EMA21, and 6h close > 6h open (bullish candle).
# Short: 12h ADX > 25, 6h close < 6h EMA21, and 6h close < 6h open (bearish candle).
# Exit when trend weakens (ADX < 20) or price crosses EMA21 in opposite direction.
# Designed to work in both bull and bear markets by following strong trends only.

name = "6h_12h_ADX_Trend_With_Pullback_EMA"
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
    open_price = prices['open'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (14-period)
    def smooth(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr = smooth(tr, 14)
    plus_di = 100 * smooth(plus_dm, 14) / atr
    minus_di = 100 * smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth(dx, 14)
    
    # Handle division by zero or invalid values
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    
    # Align 12h ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # 6h EMA21 for pullback entries
    close_series = pd.Series(close)
    ema_21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_21[i]) or 
            np.isnan(close[i]) or np.isnan(open_price[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        price = close[i]
        ema_val = ema_21[i]
        is_bullish_candle = close[i] > open_price[i]
        is_bearish_candle = close[i] < open_price[i]
        
        if position == 0:
            # Long: strong trend (ADX > 25), price above EMA21, bullish candle
            if (adx_val > 25 and 
                price > ema_val and
                is_bullish_candle):
                signals[i] = 0.25
                position = 1
            # Short: strong trend (ADX > 25), price below EMA21, bearish candle
            elif (adx_val > 25 and 
                  price < ema_val and
                  is_bearish_candle):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakens (ADX < 20) or price crosses below EMA21
            if adx_val < 20 or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakens (ADX < 20) or price crosses above EMA21
            if adx_val < 20 or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals