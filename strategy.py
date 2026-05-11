#!/usr/bin/env python3
name = "1d_KAMA_Trend_RSI_Filter"
timeframe = "1d"
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
    
    # KAMA calculation parameters
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder for correct calculation
    
    # Correct ER calculation: need rolling volatility
    er = np.zeros(n)
    for i in range(n):
        if i < er_period:
            er[i] = np.nan
        else:
            price_change = np.abs(close[i] - close[i-er_period])
            volatility_sum = np.sum(np.abs(np.diff(close[i-er_period+1:i+1])))
            er[i] = price_change / (volatility_sum + 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_1w = close_1w > ema20_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20, 14)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(trend_up_1w_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI < 50 (not overbought) + volume confirmation
            if (close[i] > kama[i] and 
                trend_up_1w_aligned[i] and 
                rsi[i] < 50 and
                volume[i] > 1.2 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + RSI > 50 (not oversold) + volume confirmation
            elif (close[i] < kama[i] and 
                  not trend_up_1w_aligned[i] and 
                  rsi[i] > 50 and
                  volume[i] > 1.2 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI > 70 (overbought)
            if (close[i] < kama[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI < 30 (oversold)
            if (close[i] > kama[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals