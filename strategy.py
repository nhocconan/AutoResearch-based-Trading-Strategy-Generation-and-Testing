#!/usr/bin/env python3
name = "6H_Keltner_Channel_RSI_Extreme"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA100 for trend filter
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Calculate 6h Keltner Channel components
    # ATR(10)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # EMA(20) of typical price for Keltner middle
    tp = (high + low + close) / 3
    ema20_tp = pd.Series(tp).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    keltner_upper = ema20_tp + 2.0 * atr10
    keltner_lower = ema20_tp - 2.0 * atr10
    
    # Calculate 6h RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema100_1d_aligned[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above daily EMA100
        uptrend = close[i] > ema100_1d_aligned[i]
        # Downtrend: price below daily EMA100
        downtrend = close[i] < ema100_1d_aligned[i]
        
        if position == 0:
            # Enter long: Uptrend + RSI oversold (<30) + price touches lower Keltner band
            if uptrend and rsi[i] < 30 and close[i] <= keltner_lower[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + RSI overbought (>70) + price touches upper Keltner band
            elif downtrend and rsi[i] > 70 and close[i] >= keltner_upper[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought (>70) or trend turns down
            if rsi[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold (<30) or trend turns up
            if rsi[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals