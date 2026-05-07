#!/usr/bin/env python3
# 1h_TripleFilter_RSI_CCI_Trend
# Hypothesis: 1h strategy combining RSI oversold/overbought, CCI momentum, and 4h EMA trend filter. 
# Uses 4h EMA50 for trend direction to avoid counter-trend trades, RSI(14) for mean-reversion signals, 
# and CCI(20) for momentum confirmation. Volume filter ensures breakouts have conviction.
# Designed for low trade frequency (15-30/year) with high win rate in both bull/bear markets.
# Timeframe: 1h, uses 4h for trend filter, 1h for entry timing.

timeframe = "1h"
name = "1h_TripleFilter_RSI_CCI_Trend"
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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate CCI(20) on 1h
    tp = (high + low + close) / 3
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
    
    # Volume filter: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have EMA50, RSI, CCI data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(cci[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30), CCI bullish (> -100), volume confirmation, 4h trend bullish
            if (rsi[i] < 30 and 
                cci[i] > -100 and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70), CCI bearish (< 100), volume confirmation, 4h trend bearish
            elif (rsi[i] > 70 and 
                  cci[i] < 100 and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: RSI overbought (>70) or trend turns bearish
            if rsi[i] > 70 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI oversold (<30) or trend turns bullish
            if rsi[i] < 30 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals