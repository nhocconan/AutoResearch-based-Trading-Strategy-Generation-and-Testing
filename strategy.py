#!/usr/bin/env python3
name = "6h_RSI20_WeeklyTrend_ObvMomentum"
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
    volume = prices['volume'].values
    
    # Weekly EMA for trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily OBV momentum (1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    # Calculate OBV
    price_change = np.diff(close_1d, prepend=close_1d[0])
    obv = np.cumsum(np.where(price_change > 0, volume_1d, np.where(price_change < 0, -volume_1d, 0)))
    # 10-period EMA of OBV
    obv_ema = pd.Series(obv).ewm(span=10, adjust=False, min_periods=10).mean().values
    obv_ema_aligned = align_htf_to_ltf(prices, df_1d, obv_ema)
    
    # 6h RSI(20)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(obv_ema_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + weekly uptrend + OBV rising
            if (rsi[i] < 30 and 
                close[i] > ema_20_1w_aligned[i] and 
                obv_ema_aligned[i] > obv_ema_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + weekly downtrend + OBV falling
            elif (rsi[i] > 70 and 
                  close[i] < ema_20_1w_aligned[i] and 
                  obv_ema_aligned[i] < obv_ema_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 70 or weekly trend turns down
            if rsi[i] > 70 or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 30 or weekly trend turns up
            if rsi[i] < 30 or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals