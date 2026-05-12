#!/usr/bin/env python3
# 1d_WilliamsAlligator_1wTrend
# Hypothesis: Use Williams Alligator for trend detection on 1d, filtered by 1w trend direction.
# Long when Alligator jaws above teeth and lips (bullish alignment) and weekly close > weekly EMA50.
# Short when jaws below teeth and lips (bearish alignment) and weekly close < weekly EMA50.
# Exit when alignment breaks or weekly trend reverses. Williams Alligator captures sustained trends with fewer whipsaws.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe momentum, improving win rate in both bull and bear markets.

name = "1d_WilliamsAlligator_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator on 1d: SMMA (Smoothed Moving Average)
    # Jaws: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(data, period, shift):
        # Smoothed Moving Average: EMA-like but with smoothing
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        # Shift the result to the right by 'shift' bars
        if shift > 0:
            result = np.roll(result, shift)
            result[:shift] = np.nan
        return result
    
    jaws = smma(close, 13, 8)
    teeth = smma(close, 8, 5)
    lips = smma(close, 5, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get weekly close aligned to daily for trend comparison
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        weekly_close_current = close_1w_aligned[i]
        
        trend_up = weekly_close_current > ema50_1w_aligned[i]
        trend_down = weekly_close_current < ema50_1w_aligned[i]
        
        # Alligator alignment: jaws > teeth > lips = bullish, jaws < teeth < lips = bearish
        bullish_alignment = jaws[i] > teeth[i] and teeth[i] > lips[i]
        bearish_alignment = jaws[i] < teeth[i] and teeth[i] < lips[i]
        
        if position == 0:
            # LONG: bullish Alligator alignment AND weekly uptrend
            if bullish_alignment and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish Alligator alignment AND weekly downtrend
            elif bearish_alignment and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks OR weekly trend turns down
            if not bullish_alignment or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks OR weekly trend turns up
            if not bearish_alignment or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals