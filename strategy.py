#!/usr/bin/env python3
# 1d_Keltner_Breakout_Trend_Momentum
# Hypothesis: On 1d timeframe, buy when price breaks above Keltner upper band with bullish momentum (MACD > 0),
# sell when price breaks below Keltner lower band with bearish momentum (MACD < 0).
# Uses 1w EMA50 as trend filter: only take longs in uptrend (price > weekly EMA50),
# only take shorts in downtrend (price < weekly EMA50).
# Designed for 1d to achieve 7-25 trades/year with low turnover and high edge.

name = "1d_Keltner_Breakout_Trend_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Keltner Channel (20, 10) on 1d
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA20 of typical price
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    ema_tp_20 = pd.Series(tp_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR10 of 1d
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(low_1d[1:], high_1d[:-1])
    tr1 = np.concatenate([[np.nan], tr1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr2 = np.concatenate([[np.nan], tr2])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr3 = np.concatenate([[np.nan], tr3])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    kel_upper = ema_tp_20 + 2 * atr_10
    kel_lower = ema_tp_20 - 2 * atr_10
    
    # MACD (12,26,9) on 1d close
    ema_fast = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Align 1w trend to 1d
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align 1d indicators to 1d (no alignment needed as already 1d, but for consistency)
    kel_upper_aligned = align_htf_to_ltf(prices, df_1d, kel_upper)
    kel_lower_aligned = align_htf_to_ltf(prices, df_1d, kel_lower)
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(kel_upper_aligned[i]) or \
           np.isnan(kel_lower_aligned[i]) or np.isnan(macd_hist_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        close_1w_series = pd.Series(close_1w)
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w_series.values)
        is_uptrend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        is_downtrend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Keltner upper + bullish MACD + uptrend
            if close[i] > kel_upper_aligned[i] and macd_hist_aligned[i] > 0 and is_uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Keltner lower + bearish MACD + downtrend
            elif close[i] < kel_lower_aligned[i] and macd_hist_aligned[i] < 0 and is_downtrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below Keltner lower or trend turns down
            if close[i] < kel_lower_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above Keltner upper or trend turns up
            if close[i] > kel_upper_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals