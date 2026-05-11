#!/usr/bin/env python3
name = "1d_1w_RSI_Divergence_Signal"
timeframe = "1d"
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
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for long-term trend
    weekly_close = df_1w['close'].values
    ema200_w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_trend = weekly_close > ema200_w  # True for uptrend
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume average (20-period)
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Align weekly trend and daily indicators
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    rsi_aligned = align_htf_to_ltf(prices, None, rsi)  # Same timeframe, no alignment needed
    vol_ma20_aligned = align_htf_to_ltf(prices, None, vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + RSI oversold (<30) + volume spike
            if (weekly_trend_aligned[i] and 
                rsi_aligned[i] < 30 and 
                volume[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + RSI overbought (>70) + volume spike
            elif (not weekly_trend_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  volume[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down OR RSI overbought (>70)
            if (not weekly_trend_aligned[i] or rsi_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up OR RSI oversold (<30)
            if (weekly_trend_aligned[i] or rsi_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals