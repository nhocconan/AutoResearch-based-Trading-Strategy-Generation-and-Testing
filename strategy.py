#!/usr/bin/env python3
name = "6h_RVI_Trend_12hTrend_Filter"
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
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # RVI (Relative Vigor Index) calculation
    numerator = close - open_
    denominator = high - low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    rvi_raw = numerator / denominator
    
    # Smooth RVI using EMA(10)
    rvi = pd.Series(rvi_raw).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # RVI signal line (EMA of RVI)
    rvi_signal = pd.Series(rvi).ewm(span=4, adjust=False, min_periods=4).mean().values
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # ensure RVI and other indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rvi[i]) or np.isnan(rvi_signal[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RVI crosses above signal line + above 12h EMA50 + volume confirmation
            if (rvi[i] > rvi_signal[i] and rvi[i-1] <= rvi_signal[i-1] and 
                close[i] > ema_50_12h_aligned[i] and vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: RVI crosses below signal line + below 12h EMA50 + volume confirmation
            elif (rvi[i] < rvi_signal[i] and rvi[i-1] >= rvi_signal[i-1] and 
                  close[i] < ema_50_12h_aligned[i] and vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RVI crosses below signal line
            if rvi[i] < rvi_signal[i] and rvi[i-1] >= rvi_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RVI crosses above signal line
            if rvi[i] > rvi_signal[i] and rvi[i-1] <= rvi_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals