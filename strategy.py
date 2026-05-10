#!/usr/bin/env python3
# 1d_WMA_Crossover_1wTrend_Volume
# Hypothesis: Uses weekly trend filter with daily WMA crossover for entries. 
# Long when price crosses above 50-day WMA with volume confirmation and weekly uptrend (price > 200-week WMA).
# Short when price crosses below 50-day WMA with volume confirmation and weekly downtrend (price < 200-week WMA).
# Exit on opposite crossover. Designed for low trade frequency with trend-following edge in both bull/bear markets.
# Targets 10-25 trades per year on 1d timeframe with position size 0.25.

name = "1d_WMA_Crossover_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def wma(arr, period):
    """Calculate Weighted Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    weights = np.arange(1, period + 1)
    return np.convolve(arr, weights[::-1], mode='full')[:len(arr)] / weights.sum()

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 200-week WMA for trend
    wma_200_1w = wma(df_1w['close'].values, 200)
    wma_200_1w_aligned = align_htf_to_ltf(prices, df_1w, wma_200_1w)
    
    # Calculate daily 50-period WMA for entry signal
    wma_50 = wma(close, 50)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for WMA(50)
    
    for i in range(start_idx, n):
        if np.isnan(wma_200_1w_aligned[i]) or np.isnan(wma_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # WMA crossover signals
        price_above_wma50 = close[i] > wma_50[i]
        price_below_wma50 = close[i] < wma_50[i]
        was_above_wma50 = close[i-1] > wma_50[i-1] if i > 0 else False
        was_below_wma50 = close[i-1] < wma_50[i-1] if i > 0 else False
        
        bullish_cross = price_above_wma50 and was_below_wma50
        bearish_cross = price_below_wma50 and was_above_wma50
        
        # Trend filter: price relative to 200-week WMA
        price_above_weekly_trend = close[i] > wma_200_1w_aligned[i]
        price_below_weekly_trend = close[i] < wma_200_1w_aligned[i]
        
        if position == 0:
            # Long entry: bullish WMA crossover with volume confirmation and weekly uptrend
            if bullish_cross and volume_confirm[i] and price_above_weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish WMA crossover with volume confirmation and weekly downtrend
            elif bearish_cross and volume_confirm[i] and price_below_weekly_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish WMA crossover
            if bearish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish WMA crossover
            if bullish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals