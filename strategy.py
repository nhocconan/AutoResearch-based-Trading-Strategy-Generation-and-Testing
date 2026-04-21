#!/usr/bin/env python3
"""
12h_1d_Engulfing_Reversal_V1
Hypothesis: On 12h timeframe, bullish/bearish engulfing patterns with volume confirmation
and aligned 1d trend (EMA50) yield high-probability reversals. Engulfing patterns signal
strong momentum shifts; volume confirms institutional participation; EMA50 filter ensures
trades align with higher-timeframe trend, reducing whipsaws in ranging markets. Designed
for low trade frequency (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend and engulfing detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Align daily OHLC to 12h timeframe for engulfing detection
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_arr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if EMA not ready
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Use prior completed day's OHLC for engulfing pattern
        prev_open = open_1d_aligned[i-1]
        prev_close = close_1d_aligned[i-1]
        curr_open = open_1d_aligned[i]
        curr_close = close_1d_aligned[i]
        
        # Bullish engulfing: current green candle fully engulfs prior red candle
        bullish_engulf = (curr_close > curr_open) and (prev_close < prev_open) and \
                         (curr_open <= prev_close) and (curr_close >= prev_open)
        # Bearish engulfing: current red candle fully engulfs prior green candle
        bearish_engulf = (curr_close < curr_open) and (prev_close > prev_open) and \
                         (curr_open >= prev_close) and (curr_close <= prev_open)
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = prices['volume'].iloc[i] > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price > EMA50 for long, price < EMA50 for short
        price = prices['close'].iloc[i]
        trend_long = price > ema_50_1d_aligned[i]
        trend_short = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: bullish engulfing + volume confirmation + uptrend
            if bullish_engulf and volume_ok and trend_long:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing + volume confirmation + downtrend
            elif bearish_engulf and volume_ok and trend_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish engulfing or trend turns bearish
            if bearish_engulf or not trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish engulfing or trend turns bullish
            if bullish_engulf or not trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Engulfing_Reversal_V1"
timeframe = "12h"
leverage = 1.0