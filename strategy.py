#!/usr/bin/env python3
# 1h_4H1D_Trend_Filter_with_1H_RSI_Entry
# Hypothesis: Use 4h and 1d trend alignment (same direction) as directional filter.
# Enter long only when both 4h and 1d are in uptrend, short when both in downtrend.
# Use 1h RSI for entry timing (oversold for long, overbought for short) to catch pullbacks.
# Exit when trend alignment breaks or RSI reaches opposite extreme.
# Designed for 1h timeframe with low trade frequency (<30/year) to avoid fee drag.
# Works in bull/bear markets by requiring multi-timeframe trend consensus.

name = "1h_4H1D_Trend_Filter_with_1H_RSI_Entry"
timeframe = "1h"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA50 trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    
    # 1d EMA50 trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 4h and 1d trends to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 1h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if 4h and 1d trends are aligned
        trend_aligned_up = trend_4h_up_aligned[i] > 0.5 and trend_1d_up_aligned[i] > 0.5
        trend_aligned_down = trend_4h_down_aligned[i] > 0.5 and trend_1d_down_aligned[i] > 0.5
        
        if position == 0:
            # Long: both 4h and 1d uptrend + 1h RSI oversold
            if trend_aligned_up and rsi[i] < 30:
                signals[i] = 0.20
                position = 1
            # Short: both 4h and 1d downtrend + 1h RSI overbought
            elif trend_aligned_down and rsi[i] > 70:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: trend alignment breaks or RSI overbought
            if not trend_aligned_up or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: trend alignment breaks or RSI oversold
            if not trend_aligned_down or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals