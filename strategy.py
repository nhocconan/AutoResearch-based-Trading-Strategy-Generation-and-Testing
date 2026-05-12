#!/usr/bin/env python3
name = "4h_RSI_2_Trend_Stop"
timeframe = "4h"
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
    
    # Daily EMA200 for trend filter
    df_1d = get_hef_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(2) on 4h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 10 and price > EMA200 (oversold in uptrend)
            if rsi[i] < 10 and close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 90 and price < EMA200 (overbought in downtrend)
            elif rsi[i] > 90 and close[i] < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 60 or price < EMA200
            if rsi[i] > 60 or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 40 or price > EMA200
            if rsi[i] < 40 or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals