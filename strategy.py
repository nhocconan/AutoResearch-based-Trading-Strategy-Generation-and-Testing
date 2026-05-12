#!/usr/bin/env python3
name = "4h_KAMA_Direction_Trend_Momentum_v1"
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
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # KAMA on 4h (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    vol = np.abs(np.diff(close, prepend=close[0]))
    er = np.where(vol != 0, change / vol, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Momentum: ROC(10)
    roc = np.zeros_like(close)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(roc[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA + above 1d EMA34 + positive ROC + volume filter
            if close[i] > kama[i] and close[i] > ema_34_1d_aligned[i] and roc[i] > 0 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + below 1d EMA34 + negative ROC + volume filter
            elif close[i] < kama[i] and close[i] < ema_34_1d_aligned[i] and roc[i] < 0 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA or below 1d EMA34
            if close[i] < kama[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA or above 1d EMA34
            if close[i] > kama[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals