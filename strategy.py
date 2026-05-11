#!/usr/bin/env python3
name = "6h_Keltner_RSI_Divergence_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # EMA20 for Keltner center
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10) for Keltner width
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.concatenate([[np.nan], tr1])
    atr10 = pd.Series(tr1).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    upper = ema20 + 2 * atr10
    lower = ema20 - 2 * atr10
    
    # RSI(14)
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily trend filter: price above/below daily EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            if i > 0:
                vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(rsi[i]) or
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches lower Keltner + RSI < 40 (oversold) + daily uptrend + volume confirmation
            if (low[i] <= lower[i] and 
                rsi[i] < 40 and 
                trend_up_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches upper Keltner + RSI > 60 (overbought) + daily downtrend + volume confirmation
            elif (high[i] >= upper[i] and 
                  rsi[i] > 60 and 
                  not trend_up_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches upper Keltner or RSI > 70 or trend changes
            if (high[i] >= upper[i] or rsi[i] > 70 or not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches lower Keltner or RSI < 30 or trend changes
            if (low[i] <= lower[i] or rsi[i] < 30 or trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals