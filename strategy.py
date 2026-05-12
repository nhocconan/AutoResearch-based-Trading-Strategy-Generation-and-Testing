#!/usr/bin/env python3
name = "6h_RelativeStrengthIndex_WeeklyTrendFilter"
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
    
    # Load weekly data for trend filter (EMA14)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_14_1w = pd.Series(close_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    ema_14_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_14_1w)
    
    # RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_14_1w_aligned[i]) or 
            np.isnan(rsi_values[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 + above weekly EMA14 + volume filter
            if rsi_values[i] < 30 and close[i] > ema_14_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 + below weekly EMA14 + volume filter
            elif rsi_values[i] > 70 and close[i] < ema_14_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 70 or below weekly EMA14
            if rsi_values[i] > 70 or close[i] < ema_14_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 30 or above weekly EMA14
            if rsi_values[i] < 30 or close[i] > ema_14_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals