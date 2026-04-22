#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend following with 4h/1d EMA alignment filter and 4-hour breakout entry
# - Uses 4h and 1d EMA(50/100) to determine trend direction (both must agree)
# - Enters on 1h breakouts of the prior 4-hour high/low in the direction of the trend
# - Exits when trend disagreement occurs or opposite signal appears
# - Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years)
# - Uses discrete position size of 0.20 to minimize fee churn

def generate_signals(prices):
    n = len(prices)
    if n < 120:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h Trend Indicators (EMA 50/100) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 100:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_4h = pd.Series(close_4h).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_100_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_100_4h)
    
    # === 1d Trend Indicators (EMA 50/100) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # === 1h Indicators ===
    ema_50_1h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4-hour breakout levels (using prior 4 completed candles)
    max_high_4h = pd.Series(high).rolling(window=4, min_periods=4).max().shift(1).values
    min_low_4h = pd.Series(low).rolling(window=4, min_periods=4).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are valid (100 for EMA + 4 for lookback + 1 shift)
    start_idx = 105
    for i in range(start_idx, n):
        # Skip if any indicator data is missing
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_100_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_100_1d_aligned[i]) or
            np.isnan(ema_50_1h[i]) or np.isnan(max_high_4h[i]) or np.isnan(min_low_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend alignment
        four_h_up = ema_50_4h_aligned[i] > ema_100_4h_aligned[i]
        four_h_down = ema_50_4h_aligned[i] < ema_100_4h_aligned[i]
        one_d_up = ema_50_1d_aligned[i] > ema_100_1d_aligned[i]
        one_d_down = ema_50_1d_aligned[i] < ema_100_1d_aligned[i]
        
        # Entry conditions: breakout + trend alignment + close filter
        long_breakout = high[i] > max_high_4h[i]
        short_breakout = low[i] < min_low_4h[i]
        long_condition = four_h_up and one_d_up and long_breakout and (close[i] > ema_50_1h[i])
        short_condition = four_h_down and one_d_down and short_breakout and (close[i] < ema_50_1h[i])
        
        # Exit conditions: trend disagreement or opposite signal
        if position == 0:
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:  # Long position
            if short_condition or not (four_h_up and one_d_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            if long_condition or not (four_h_down and one_d_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_EMA50_100_Trend_Breakout"
timeframe = "1h"
leverage = 1.0