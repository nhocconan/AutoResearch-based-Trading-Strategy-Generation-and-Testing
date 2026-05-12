#!/usr/bin/env python3
"""
6h_1d_VolumeWeighted_Candlestick_Engulfing
Hypothesis: Combining daily volume-weighted RSI with engulfing candlestick patterns on 6h provides mean-reversion entries during overextended moves while avoiding chop. The volume-weighted RSI identifies exhaustion, and engulfing candles confirm reversal. Works in both bull (buy dips) and bear (sell rallies) by fading extremes. Target: 15-25 trades/year per symbol.
"""

name = "6h_1d_VolumeWeighted_Candlestick_Engulfing"
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
    
    # Volume-weighted RSI on daily timeframe (more responsive to institutional activity)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate price changes
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted average gain/loss
    vol_weighted_gain = gain * volume_1d
    vol_weighted_loss = loss * volume_1d
    
    # Wilder smoothing with volume weighting
    avg_gain = pd.Series(vol_weighted_gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(vol_weighted_loss).ewm(alpha=1/14, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Engulfing pattern detection on 6h
    bullish_engulf = (close > open_) & (open_ < close_) & (close > open_.shift(1)) & (open_ < close_.shift(1))
    bearish_engulf = (close < open_) & (open_ > close_) & (close < open_.shift(1)) & (open_ > close_.shift(1))
    
    # Need open prices for engulfing - reconstruct from available data
    open_prices = prices['open'].values
    bullish_engulf = (close > open_prices) & (open_prices < close) & (close > np.roll(open_prices, 1)) & (open_prices < np.roll(close, 1))
    bearish_engulf = (close < open_prices) & (open_prices > close) & (close < np.roll(open_prices, 1)) & (open_prices > np.roll(close, 1))
    
    # Handle first element
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    # Align daily RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Oversold RSI + bullish engulfing candle
            if (rsi_1d_aligned[i] < 30) and bullish_engulf[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought RSI + bearish engulfing candle
            elif (rsi_1d_aligned[i] > 70) and bearish_engulf[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral or bearish engulf
            if (rsi_1d_aligned[i] >= 50) or bearish_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral or bullish engulf
            if (rsi_1d_aligned[i] <= 50) or bullish_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals