#!/usr/bin/env python3
# 1d_1W_Camarilla_R1_S1_Breakout_Trend_Filter
# Hypothesis: Trade Camarilla pivot level breaks on daily timeframe with weekly trend filter.
# Long when price breaks above R1 with weekly uptrend, short when breaks below S1 with weekly downtrend.
# Uses volume confirmation to avoid false breakouts. Designed for low frequency (~15 trades/year) 
# to minimize fee drag and work in both bull and bear markets by following the weekly trend.

name = "1d_1W_Camarilla_R1_S1_Breakout_Trend_Filter"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = close_1w > ema20_1w
    trend_1w_down = close_1w < ema20_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Daily Camarilla pivot levels (based on previous day)
    # Calculate pivot points using previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first value
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.1 / 12)
    S1 = pivot - (range_val * 1.1 / 12)
    
    # Volume average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(vol_ma[i]) or
            np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 with weekly uptrend and volume confirmation
            if (close[i] > R1[i] and 
                trend_1w_up_aligned[i] > 0.5 and
                volume[i] > vol_ma[i] * 1.5):  # 50% above average volume
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with weekly downtrend and volume confirmation
            elif (close[i] < S1[i] and 
                  trend_1w_down_aligned[i] > 0.5 and
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below pivot or weekly trend changes
            if (close[i] < pivot[i] or 
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above pivot or weekly trend changes
            if (close[i] > pivot[i] or 
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals