#!/usr/bin/env python3
"""
4h_RSI_Engulfing_Filter
Hypothesis: RSI(14) divergence from price (oversold/overbought) combined with 
engulfing candle patterns and 1-day EMA trend filter yields high-probability 
reversals in both bull and bear markets. Low-frequency signals reduce fee drag.
"""

name = "4h_RSI_Engulfing_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Bullish/Bearish Engulfing
    bull_engulf = (close > open_) & (open_ < close_) & (close > open_.shift(1)) & (open_ < close_.shift(1))
    bear_engulf = (close < open_) & (open_ > close_) & (close < open_.shift(1)) & (open_ > close_.shift(1))
    # Fix: define open_
    open_ = prices['open'].values
    bull_engulf = (close > open_) & (open_ < close_) & (close > np.roll(open_, 1)) & (open_ < np.roll(close_, 1)) if 'close_' in locals() else (close > open_) & (open_ < close_) & (close > np.roll(open_, 1)) & (open_ < np.roll(close_, 1))
    bear_engulf = (close < open_) & (open_ > close_) & (close < np.roll(open_, 1)) & (open_ > np.roll(close_, 1))
    # Recompute with correct variables
    open_ = prices['open'].values
    close_ = close  # alias
    bull_engulf = (close > open_) & (open_ < close_) & (close > np.roll(open_, 1)) & (open_ < np.roll(close_, 1))
    bear_engulf = (close < open_) & (open_ > close_) & (close < np.roll(open_, 1)) & (open_ > np.roll(close_, 1))
    
    # Handle first element
    bull_engulf[0] = False
    bear_engulf[0] = False
    
    # 1-day EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    uptrend_1d = close > ema_200_1d_aligned
    downtrend_1d = close < ema_200_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # wait for EMA200 warmup
        if position == 0:
            # LONG: RSI < 30 (oversold) + bullish engulfing + uptrend on 1D
            if rsi[i] < 30 and bull_engulf[i] and uptrend_1d[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) + bearish engulfing + downtrend on 1D
            elif rsi[i] > 70 and bear_engulf[i] and downtrend_1d[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI > 50 or bearish engulfing
            if rsi[i] > 50 or bear_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 or bullish engulfing
            if rsi[i] < 50 or bull_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals