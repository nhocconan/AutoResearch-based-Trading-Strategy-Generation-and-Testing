#!/usr/bin/env python3
name = "1d_WeeklyKeltnerBreakout_TrendFilter"
timeframe = "1d"
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
    
    # Weekly ATR for Keltner channels (20-period EMA)
    df_1w = get_htf_data(prices, '1w')
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Calculate True Range for weekly
    tr1 = high_w - low_w
    tr2 = np.abs(high_w - np.concatenate([[close_w[0]], close_w[:-1]]))
    tr3 = np.abs(low_w - np.concatenate([[close_w[0]], close_w[:-1]]))
    tr_w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_w = pd.Series(tr_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly EMA (middle of Keltner)
    ema_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly Keltner channels
    upper_w = ema_w + 2.0 * atr_w
    lower_w = ema_w - 2.0 * atr_w
    
    # Align to daily
    ema_w_aligned = align_htf_to_ltf(prices, df_1w, ema_w)
    upper_w_aligned = align_htf_to_ltf(prices, df_1w, upper_w)
    lower_w_aligned = align_htf_to_ltf(prices, df_1w, lower_w)
    
    # Weekly trend (price above/below EMA)
    weekly_trend_up = close_w > ema_w
    weekly_trend_down = close_w < ema_w
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Daily volume filter: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_w_aligned[i]) or np.isnan(upper_w_aligned[i]) or np.isnan(lower_w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + price breaks above upper Keltner + volume filter
            if (weekly_trend_up_aligned[i] and 
                close[i] > upper_w_aligned[i] and 
                volume[i] > vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price breaks below lower Keltner + volume filter
            elif (weekly_trend_down_aligned[i] and 
                  close[i] < lower_w_aligned[i] and 
                  volume[i] > vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA OR trend changes
            if close[i] < ema_w_aligned[i] or not weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly EMA OR trend changes
            if close[i] > ema_w_aligned[i] or not weekly_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals