#!/usr/bin/env python3
name = "6h_RSI_Bollinger_Band_Squeeze"
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
    
    # 1d Bollinger Bands: middle=20, std=2
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma10 = pd.Series(bb_width).rolling(window=10, min_periods=10).mean().values
    bb_width_ma10_aligned = align_htf_to_ltf(prices, df_1d, bb_width_ma10)
    bb_width_threshold = 0.05  # Squeeze threshold
    
    # 6h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h Bollinger Bands for entry/exit
    bb_middle_6h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std_6h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper_6h = bb_middle_6h + 2 * bb_std_6h
    bb_lower_6h = bb_middle_6h - 2 * bb_std_6h
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for BB and RSI
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or np.isnan(bb_width_ma10_aligned[i]) or
            np.isnan(bb_upper_6h[i]) or np.isnan(bb_lower_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Bollinger Squeeze condition: low volatility
        squeeze = bb_width_ma10_aligned[i] < bb_width_threshold
        
        if position == 0:
            # Long: RSI < 30 (oversold) + squeeze + price at lower band
            if rsi[i] < 30 and squeeze and close[i] <= bb_lower_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + squeeze + price at upper band
            elif rsi[i] > 70 and squeeze and close[i] >= bb_upper_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 50 or price at upper band
            if rsi[i] > 50 or close[i] >= bb_upper_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 or price at lower band
            if rsi[i] < 50 or close[i] <= bb_lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals