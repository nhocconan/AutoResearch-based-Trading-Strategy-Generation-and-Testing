#!/usr/bin/env python3
name = "4h_RSI20_LongOnly_With_Momentum_Filter"
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
    
    # Daily trend: 50-day EMA
    df_1d = get_he_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    trend_up = close > ema_50_1d_aligned
    
    # Daily volume filter: volume > 1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Momentum filter: ROC(5) > 0
    roc5 = np.zeros_like(close)
    roc5[5:] = (close[5:] - close[:-5]) / close[:-5] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(roc5[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 20 (oversold) + positive ROC(5) + daily uptrend + volume filter
            if rsi[i] < 20 and roc5[i] > 0 and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit: RSI > 70 (overbought) or trend down
            if rsi[i] > 70 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals