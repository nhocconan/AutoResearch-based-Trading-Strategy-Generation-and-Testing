#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_Hybrid_Signal_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 4h data for trend and volatility filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for session and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h ATR(14) for volatility filter
    tr_4h = np.maximum(df_4h['high'].values - df_4h['low'].values,
                       np.maximum(np.abs(df_4h['high'].values - np.roll(df_4h['close'].values, 1)),
                                  np.absolute(np.roll(df_4h['close'].values, 1) - df_4h['low'].values)))
    tr_4h[0] = df_4h['high'].values[0] - df_4h['low'].values[0]
    atr14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr14_4h)
    
    # 1d ATR(10) for volatility regime filter (high vol = trend following, low vol = mean reversion)
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                  np.absolute(np.roll(df_1d['close'].values, 1) - df_1d['low'].values)))
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Price position relative to 4h EMA50
    price_vs_ema = close - ema50_4h_aligned
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 14, 10)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(atr14_4h_aligned[i]) or
            np.isnan(atr10_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip outside session
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio = atr10_1d_aligned[i] / (atr14_4h_aligned[i] + 1e-10)
        price_above_ema = price_vs_ema[i] > 0
        
        if position == 0:
            # Trend following in high volatility regimes (ATR ratio > 1.2)
            if atr_ratio > 1.2:
                if price_above_ema and close[i] > close[i-1]:
                    signals[i] = 0.20
                    position = 1
                elif not price_above_ema and close[i] < close[i-1]:
                    signals[i] = -0.20
                    position = -1
            # Mean reversion in low volatility regimes (ATR ratio <= 1.2)
            else:
                if not price_above_ema and close[i] < close[i-1]:
                    signals[i] = 0.20
                    position = 1
                elif price_above_ema and close[i] > close[i-1]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA or volatility regime shifts
            if not price_above_ema or atr_ratio > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above EMA or volatility regime shifts
            if price_above_ema or atr_ratio > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals