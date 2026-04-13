# 1d_1w_HTF_Trend_Filtered_Trend
# Hypothesis: Use weekly EMA cross (21/55) as primary trend filter on daily chart.
# Enter long when daily close > weekly EMA21 and daily EMA21 crosses above EMA55.
# Enter short when daily close < weekly EMA21 and daily EMA21 crosses below EMA55.
# Exit on opposite signal or weekly EMA cross reversal.
# This reduces whipsaw by requiring alignment between daily momentum and weekly trend.
# Target: 15-25 trades/year per symbol, works in bull via trend continuation and
# in bear via trend reversals confirmed by higher timeframe.

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Calculate weekly EMA21 and EMA55
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55_1w = pd.Series(close_1w).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    ema55_1w_aligned = align_htf_to_ltf(prices, df_1w, ema55_1w)
    
    # Calculate daily EMAs for entry timing
    ema21_daily = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55_daily = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any required data is not ready
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(ema55_1w_aligned[i]) or
            np.isnan(ema21_daily[i]) or np.isnan(ema55_daily[i])):
            signals[i] = 0.0
            continue
        
        # Check weekly trend alignment
        weekly_uptrend = ema21_1w_aligned[i] > ema55_1w_aligned[i]
        weekly_downtrend = ema21_1w_aligned[i] < ema55_1w_aligned[i]
        
        # Daily EMA cross signals
        daily_bullish_cross = ema21_daily[i] > ema55_daily[i] and ema21_daily[i-1] <= ema55_daily[i-1]
        daily_bearish_cross = ema21_daily[i] < ema55_daily[i] and ema21_daily[i-1] >= ema55_daily[i-1]
        
        # Long: weekly uptrend + daily bullish cross
        if weekly_uptrend and daily_bullish_cross and position != 1:
            position = 1
            signals[i] = position_size
        # Short: weekly downtrend + daily bearish cross
        elif weekly_downtrend and daily_bearish_cross and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit: weekly trend reversal or opposite daily cross
        elif position == 1 and (not weekly_uptrend or daily_bearish_cross):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not weekly_downtrend or daily_bullish_cross):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_HTF_Trend_Filtered_Trend"
timeframe = "1d"
leverage = 1.0