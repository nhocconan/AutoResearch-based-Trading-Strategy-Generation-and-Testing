#!/usr/bin/env python3
# 6H_1W_1D_OrderBlock_Reversal_Trend
# Hypothesis: On 6h timeframe, enter long when price closes above a prior weekly demand zone (bullish order block) with 1d uptrend and volume confirmation.
# Enter short when price closes below a prior weekly supply zone (bearish order block) with 1d downtrend and volume confirmation.
# Uses weekly order blocks as institutional supply/demand zones, filtered by 1d trend and volume to avoid false breakouts.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) with high win rate in both bull and bear markets.

name = "6H_1W_1D_OrderBlock_Reversal_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for order blocks
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Identify weekly bullish order blocks (demand zones): 
    # A bullish OB is the last down candle before a strong up move
    # We define it as: when weekly close > weekly open and the prior candle was bearish
    ob_bullish = np.zeros(len(df_1w), dtype=bool)
    ob_bearish = np.zeros(len(df_1w), dtype=bool)
    
    for i in range(1, len(df_1w)):
        # Bullish OB: current candle bullish and previous candle bearish
        if close_1w[i] > high_1w[i-1] and close_1w[i-1] < high_1w[i-1]:  # gapped up from bearish candle
            ob_bullish[i-1] = True  # mark the bearish candle as OB
        # Bearish OB: current candle bearish and previous candle bullish
        elif close_1w[i] < low_1w[i-1] and close_1w[i-1] > low_1w[i-1]:  # gapped down from bullish candle
            ob_bearish[i-1] = True  # mark the bullish candle as OB
    
    # Alternative simpler method: use swing points
    # Bullish OB: lowest low before a strong up move
    # Bearish OB: highest high before a strong down move
    # Reset arrays
    ob_bullish = np.zeros(len(df_1w), dtype=bool)
    ob_bearish = np.zeros(len(df_1w), dtype=bool)
    
    # Find swing lows and highs
    for i in range(2, len(df_1w)-2):
        # Swing low: lowest low in 5-bar window
        if low_1w[i] == min(low_1w[i-2:i+3]):
            # Check if followed by strong up move (next 2 bars close higher)
            if i+2 < len(df_1w) and close_1w[i+1] > close_1w[i] and close_1w[i+2] > close_1w[i]:
                ob_bullish[i] = True
        # Swing high: highest high in 5-bar window
        if high_1w[i] == max(high_1w[i-2:i+3]):
            # Check if followed by strong down move (next 2 bars close lower)
            if i+2 < len(df_1w) and close_1w[i+1] < close_1w[i] and close_1w[i+2] < close_1w[i]:
                ob_bearish[i] = True
    
    # Create OB level arrays (price levels)
    ob_bullish_level = np.where(ob_bullish, low_1w, np.nan)
    ob_bearish_level = np.where(ob_bearish, high_1w, np.nan)
    
    # Forward fill OB levels to create zones
    ob_bullish_level_series = pd.Series(ob_bullish_level)
    ob_bearish_level_series = pd.Series(ob_bearish_level)
    ob_bullish_level_ffilled = ob_bullish_level_series.ffill().values
    ob_bearish_level_ffilled = ob_bearish_level_series.ffill().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d trend: EMA(34) on close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema_34
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align weekly and daily indicators to 6h
    ob_bullish_aligned = align_htf_to_ltf(prices, df_1w, ob_bullish_level_ffilled)
    ob_bearish_aligned = align_htf_to_ltf(prices, df_1w, ob_bearish_level_ffilled)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ob_bullish_aligned[i]) or np.isnan(ob_bearish_aligned[i]) or 
            np.isnan(trend_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price closes above weekly bullish OB + 1d uptrend + volume confirmation
            if (close[i] > ob_bullish_aligned[i] and trend_up_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price closes below weekly bearish OB + 1d downtrend + volume confirmation
            elif (close[i] < ob_bearish_aligned[i] and not trend_up_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below weekly bearish OB or trend changes
            if close[i] < ob_bearish_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above weekly bullish OB or trend changes
            if close[i] > ob_bullish_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals