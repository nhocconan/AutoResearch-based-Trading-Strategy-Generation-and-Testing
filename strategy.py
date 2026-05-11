#!/usr/bin/env python3
# 1d_1w_WeeklyHighLow_Breakout_Trend_Confirmation
# Hypothesis: Buy when price breaks above prior weekly high with 1w EMA50 uptrend, sell when breaks below weekly low with downtrend.
# Uses weekly timeframe for trend and structure, daily for execution. Low frequency (~10-20 trades/year) to minimize fee drag.
# Works in bull markets via breakouts and in bear via shorting breakdowns. Trend filter avoids whipsaws.

name = "1d_1w_WeeklyHighLow_Breakout_Trend_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1w Data (loaded ONCE) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly High/Low (prior week's extreme) ===
    # Use previous week's high/low to avoid look-ahead
    weekly_high = np.roll(high_1w, 1)
    weekly_low = np.roll(low_1w, 1)
    weekly_high[0] = np.nan  # first value invalid
    weekly_low[0] = np.nan
    
    # Align weekly levels to daily
    weekly_high_d = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_d = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # === 1w EMA50 Trend Filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA50)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high_d[i]) or np.isnan(weekly_low_d[i]) or 
            np.isnan(ema50_1w_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above weekly high with uptrend
            if close[i] > weekly_high_d[i] and close[i] > ema50_1w_d[i]:
                signals[i] = position_size
                position = 1
            # Short: Break below weekly low with downtrend
            elif close[i] < weekly_low_d[i] and close[i] < ema50_1w_d[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Reverse signal or trend change
            if position == 1:
                # Exit long on breakdown below weekly low OR trend turns down
                if close[i] < weekly_low_d[i] or close[i] < ema50_1w_d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short on breakout above weekly high OR trend turns up
                if close[i] > weekly_high_d[i] or close[i] > ema50_1w_d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals