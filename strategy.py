#!/usr/bin/env python3
name = "1d_WeeklyKeltner_Breakout_1wTrend_Volume"
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
    
    # Load weekly data for trend filter and ATR
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly ATR(10) for Keltner channels
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_10_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    
    # Weekly Keltner channels (EMA20 ± 2*ATR)
    keltner_upper = ema_20_1w + 2 * atr_10_1w
    keltner_lower = ema_20_1w - 2 * atr_10_1w
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)
    
    # Daily volume filter: current volume > 1.5x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above Keltner upper + above weekly EMA20 + volume filter
            if close[i] > keltner_upper_aligned[i] and close[i] > ema_20_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Keltner lower + below weekly EMA20 + volume filter
            elif close[i] < keltner_lower_aligned[i] and close[i] < ema_20_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below Keltner lower or below weekly EMA20
            if close[i] < keltner_lower_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above Keltner upper or above weekly EMA20
            if close[i] > keltner_upper_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals