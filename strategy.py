#!/usr/bin/env python3
"""
1d_Wick_Reversal_1wTrend
Hypothesis: Daily wick reversals (long wick against prior trend) with weekly trend filter and volume confirmation.
Long: long lower wick + close near high + price > weekly EMA50; Short: long upper wick + close near low + price < weekly EMA50.
Designed for low trade frequency (10-25/year) with mean-reversion in ranging markets and trend alignment in trending markets.
Works in both bull and bear via trend filter and defined risk/exit.
"""

name = "1d_Wick_Reversal_1wTrend"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Wick rejection signals
    body = np.abs(close - open_)
    range_ = high - low
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    long_lower_wick = lower_wick > 0.6 * range_
    long_upper_wick = upper_wick > 0.6 * range_
    close_near_high = close > (high - 0.3 * range_)
    close_near_low = close < (low + 0.3 * range_)
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Weekly trend filter: EMA 50
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: long lower wick, close near high, volume confirmation, uptrend filter
            if long_lower_wick[i] and close_near_high[i] and volume_confirm[i]:
                if close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: long upper wick, close near low, volume confirmation, downtrend filter
            elif long_upper_wick[i] and close_near_low[i] and volume_confirm[i]:
                if close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close below open (failed follow-through) or RSI-like exhaustion
            if close[i] < open_[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close above open (failed follow-through) or RSI-like exhaustion
            if close[i] > open_[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals