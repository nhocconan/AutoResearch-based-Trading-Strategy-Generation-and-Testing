#!/usr/bin/env python3
name = "1d_TripleBarrier_WickReversal_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Daily ATR(14) for stop distance
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.maximum(tr1, np.abs(low - np.roll(close, 1)))
    tr2[0] = high[0] - low[0]
    atr = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    
    # Daily range for wick detection
    body = np.abs(close - open_)
    total_range = high - low
    lower_wick = np.where(close >= open_, open_ - low, close - low)
    upper_wick = np.where(close <= open_, high - open_, high - close)
    
    # Bullish reversal: long lower wick, small body
    bullish_wick = (lower_wick > 2 * body) & (body < 0.3 * total_range)
    # Bearish reversal: long upper wick, small body
    bearish_wick = (upper_wick > 2 * body) & (body < 0.3 * total_range)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure ATR and weekly EMA have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(atr[i]) or np.isnan(ema10_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish weekly trend + bullish wick reversal
            if (close[i] > ema10_1w_aligned[i] and 
                bullish_wick[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish weekly trend + bearish wick reversal
            elif (close[i] < ema10_1w_aligned[i] and 
                  bearish_wick[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below entry - 1*ATR (stop) OR weekly trend turns bearish
            if (close[i] < close[i-1] - atr[i] or  # trailing stop
                close[i] < ema10_1w_aligned[i]):   # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above entry + 1*ATR (stop) OR weekly trend turns bullish
            if (close[i] > close[i-1] + atr[i] or  # trailing stop
                close[i] > ema10_1w_aligned[i]):   # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals