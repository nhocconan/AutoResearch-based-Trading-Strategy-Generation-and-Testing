#!/usr/bin/env python3
"""
1h_RSI_MeanReversion_4hTrend_Filter
Hypothesis: In choppy/range-bound markets, RSI mean reversion works when filtered by 4h trend.
Long when RSI < 30 and 4h EMA50 up; short when RSI > 70 and 4h EMA50 down.
Adds session filter (08-20 UTC) to avoid low-volume Asian session noise.
Target: 15-35 trades/year on 1h timeframe with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA50 trend filter (computed once, then aligned)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = np.full(len(close_4h), np.nan)
    
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[:50])
        k = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = close_4h[i] * k + ema50_4h[i-1] * (1 - k)
    
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Session filter: 08-20 UTC only
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # RSI ready
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold + 4h uptrend
            if (rsi[i] < 30 and 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + 4h downtrend
            elif (rsi[i] > 70 and 
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or trend breaks
            if (rsi[i] > 70 or 
                close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI oversold or trend breaks
            if (rsi[i] < 30 or 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0