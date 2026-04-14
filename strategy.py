#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA50 for trend direction
    ema50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(df_1d)):
            ema50_1d[i] = (close_1d[i] * 2 + ema50_1d[i-1] * 49) / 51
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily ATR(14) for volatility filter
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]])),
                               np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))))
    
    atr14_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr14_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr14_1d[i] = (atr14_1d[i-1] * 13 + tr[i]) / 14
    
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 1-hour Bollinger Bands (20, 2) for mean reversion entries
    bb_ma = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            bb_ma[i] = np.mean(close[i-19:i+1])
            bb_std[i] = np.std(close[i-19:i+1])
    
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or
            np.isnan(atr14_1d_aligned[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5% of price
        if atr14_1d_aligned[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        if close[i] > ema50_1d_aligned[i]:
            trend = 1  # uptrend
        else:
            trend = -1  # downtrend
        
        # Mean reversion entries with trend filter
        if position == 0:
            # Long in uptrend: price touches lower Bollinger Band
            if trend == 1 and close[i] <= bb_lower[i]:
                position = 1
                signals[i] = position_size
            # Short in downtrend: price touches upper Bollinger Band
            elif trend == -1 and close[i] >= bb_upper[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches middle Bollinger Band or trend changes
            if close[i] >= bb_ma[i] or trend == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches middle Bollinger Band or trend changes
            if close[i] <= bb_ma[i] or trend == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_DailyEMA50_BB_MeanReversion_SessionFilter"
timeframe = "1h"
leverage = 1.0