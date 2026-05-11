#!/usr/bin/env python3
name = "4h_Multi_Regime_Breakout_12h_Trend"
timeframe = "4h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 1d data for volatility and range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 12h for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate ATR(14) on 1d for volatility filter
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Calculate 20-period high/low for breakout levels
    high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period RSI for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(atr14_aligned[i]) or np.isnan(high20[i]) or np.isnan(low20[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 20-period high, above 12h EMA50, RSI > 50, volatility not extreme
            if (close[i] > high20[i] and 
                close[i] > ema50_12h_aligned[i] and 
                rsi[i] > 50 and
                atr14_aligned[i] < np.nanmedian(atr14_aligned[max(0, i-100):i+1]) * 2.0):
                signals[i] = 0.25
                position = 1
            # Short: break below 20-period low, below 12h EMA50, RSI < 50, volatility not extreme
            elif (close[i] < low20[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  rsi[i] < 50 and
                  atr14_aligned[i] < np.nanmedian(atr14_aligned[max(0, i-100):i+1]) * 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below 20-period low or below 12h EMA50
            if close[i] < low20[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above 20-period high or above 12h EMA50
            if close[i] > high20[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals