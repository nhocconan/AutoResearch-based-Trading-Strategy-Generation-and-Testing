#!/usr/bin/env python3
"""
1d_HTF_Trend_Retracement_EMA
Hypothesis: On daily timeframe, enter long when price retraces to EMA21 in weekly uptrend, short when price retraces to EMA21 in weekly downtrend. Use volume surge for confirmation. Exit when price crosses EMA50 in opposite direction. This strategy aims to capture trend continuation moves with low frequency by using higher timeframe trend and daily retracement entries, suitable for both bull and bear markets.
"""

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
    
    # Get weekly data for trend filter and EMAs
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema21_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema21_weekly)
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Weekly trend: bullish when price > EMA21, bearish when price < EMA21
    weekly_uptrend = close_weekly > ema21_weekly
    weekly_downtrend = close_weekly < ema21_weekly
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_downtrend.astype(float))
    
    # Daily EMAs for entry and exit
    ema21_daily = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_daily = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_weekly_aligned[i]) or np.isnan(ema50_weekly_aligned[i]) or
            np.isnan(ema21_daily[i]) or np.isnan(ema50_daily[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price near daily EMA21 with weekly trend alignment and volume surge
        near_ema21 = abs(close[i] - ema21_daily[i]) / ema21_daily[i] < 0.02  # Within 2% of EMA21
        long_entry = near_ema21 and weekly_uptrend_aligned[i] > 0.5 and volume_surge[i] and close[i] > ema21_daily[i]
        short_entry = near_ema21 and weekly_downtrend_aligned[i] > 0.5 and volume_surge[i] and close[i] < ema21_daily[i]
        
        # Exit when price crosses EMA50 in opposite direction
        long_exit = position == 1 and close[i] < ema50_daily[i]
        short_exit = position == -1 and close[i] > ema50_daily[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_HTF_Trend_Retracement_EMA"
timeframe = "1d"
leverage = 1.0