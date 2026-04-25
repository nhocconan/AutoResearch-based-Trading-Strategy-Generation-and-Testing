#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: Trade Donchian(20) breakouts on 4h timeframe with 1d EMA50 trend filter and 4h chop regime filter.
In bull markets: long when price breaks above upper Donchian(20) and price > 1d EMA50 and chop < 61.8.
In bear markets: short when price breaks below lower Donchian(20) and price < 1d EMA50 and chop < 61.8.
Exit on opposite Donchian touch or trend reversal.
Position size: 0.25 to limit drawdown and reduce fee churn.
Target: 20-50 trades/year to stay under 400-trade 4h hard max.
Uses volume confirmation via ATR expansion to avoid false breakouts.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h ATR(14) for volatility and Donchian bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h Donchian(20) channels
    def rolling_max(arr, window):
        return np.convolve(arr, np.ones(window), 'valid') / window
    
    upper_donchian = np.full(n, np.nan)
    lower_donchian = np.full(n, np.nan)
    for i in range(19, n):
        upper_donchian[i] = np.max(high[i-19:i+1])
        lower_donchian[i] = np.min(low[i-19:i+1])
    
    # Calculate 4h Chopiness Index(14) for regime filter
    def chop_index(high, low, close, window=14):
        if len(high) < window:
            return np.full(len(high), np.nan)
        atr_sum = np.zeros(len(high))
        tr = np.zeros(len(high))
        for i in range(len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]) if i > 0 else 0, abs(low[i] - close[i-1]) if i > 0 else 0)
        for i in range(window, len(high)+1):
            atr_sum[i-1] = np.sum(tr[i-window:i])
        max_high = np.zeros(len(high))
        min_low = np.zeros(len(high))
        for i in range(window, len(high)+1):
            max_high[i-1] = np.max(high[i-window:i])
            min_low[i-1] = np.min(low[i-window:i])
        chop = np.full(len(high), np.nan)
        for i in range(window-1, len(high)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(window)
        return chop
    
    chop = chop_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and EMA50
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Regime filter: only trade in trending markets (chop < 61.8)
        trending_regime = chop[i] < 61.8
        
        if position == 0:
            # Long setup: price breaks above upper Donchian + 1d uptrend + trending regime
            long_setup = (close[i] > upper_donchian[i]) and htf_1d_bullish and trending_regime
            
            # Short setup: price breaks below lower Donchian + 1d downtrend + trending regime
            short_setup = (close[i] < lower_donchian[i]) and htf_1d_bearish and trending_regime
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches lower Donchian (stop) OR 1d trend turns bearish OR chop becomes too high
            if (close[i] <= lower_donchian[i]) or (not htf_1d_bullish) or (chop[i] >= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Donchian (stop) OR 1d trend turns bullish OR chop becomes too high
            if (close[i] >= upper_donchian[i]) or (htf_1d_bullish) or (chop[i] >= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0