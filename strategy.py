#!/usr/bin/env python3
"""
4h_1d_RVI_Divergence_Trend
Hypothesis: Relative Vigor Index (RVI) divergence with price on 4h, confirmed by 1d trend, provides high-probability reversal signals. RVI measures trend strength; divergence signals weakening momentum before price reverses. Works in both bull and bear markets by following the higher timeframe trend. Target: 20-40 trades/year per symbol.
"""

name = "4h_1d_RVI_Divergence_Trend"
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
    
    # Calculate RVI (Relative Vigor Index)
    numerator = close - open_price
    denominator = high - low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    v = numerator / denominator
    
    # Smooth v with 4-period SMA for numerator and denominator of RVI
    v_smooth = np.zeros(n)
    for i in range(3, n):
        v_smooth[i] = np.mean(v[i-3:i+1])
    
    # Calculate RVI using 4-period SMA of v_smooth
    rvi = np.zeros(n)
    for i in range(7, n):
        num = np.sum(v_smooth[i-3:i+1])
        den = np.sum(np.abs(v_smooth[i-3:i+1]))
        if den != 0:
            rvi[i] = num / den
        else:
            rvi[i] = 0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: 34 EMA (faster for better responsiveness)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 4h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Calculate RVI slope (4-period) and price slope (4-period)
    rvi_slope = np.zeros(n)
    price_slope = np.zeros(n)
    
    for i in range(4, n):
        rvi_slope[i] = rvi[i] - rvi[i-4]
        price_slope[i] = close[i] - close[i-4]
    
    # Detect divergences
    # Bearish divergence: price makes higher high, RVI makes lower high
    bearish_divergence = (price_slope > 0) & (rvi_slope < 0)
    # Bullish divergence: price makes lower low, RVI makes higher low
    bullish_divergence = (price_slope < 0) & (rvi_slope > 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: 1d uptrend + bullish RVI divergence
            if uptrend and bullish_divergence[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + bearish RVI divergence
            elif downtrend and bearish_divergence[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 1d trend turns down or bearish divergence appears
            if not uptrend or bearish_divergence[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 1d trend turns up or bullish divergence appears
            if not downtrend or bullish_divergence[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals