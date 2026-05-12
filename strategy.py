#!/usr/bin/env python3
# 1d_Keltner_Upper_Breakout_WeeklyTrend
# Hypothesis: On 1d timeframe, enter long when price closes above Keltner upper band with weekly EMA20 uptrend.
# Enter short when price closes below Keltner lower band with weekly EMA20 downtrend.
# Exit when price crosses 20-day EMA (trend reversal).
# Uses weekly trend filter to avoid counter-trend trades and Keltner channels for volatility-based breakouts.
# Targets 15-25 trades/year for low fee drift.

name = "1d_Keltner_Upper_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-day EMA for exit
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner Channels (20, 10, 2)
    ema20_keltner = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(np.abs(high - low)).rolling(window=10, min_periods=10).mean().values
    upper_keltner = ema20_keltner + 2 * atr
    lower_keltner = ema20_keltner - 2 * atr
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 20-week EMA
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema20[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        upper_keltner_val = upper_keltner[i]
        lower_keltner_val = lower_keltner[i]
        ema20_val = ema20[i]
        weekly_trend = ema20_1w_aligned[i]
        
        if position == 0:
            # LONG: Price closes above upper Keltner with weekly uptrend
            if close[i] > upper_keltner_val and close[i] > weekly_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below lower Keltner with weekly downtrend
            elif close[i] < lower_keltner_val and close[i] < weekly_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 20-day EMA (trend reversal)
            if close[i] < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 20-day EMA (trend reversal)
            if close[i] > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals