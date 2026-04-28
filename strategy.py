#!/usr/bin/env python3
"""
1d_RSI_MeanRev_with_WeeklyTrend_Filter
Hypothesis: On daily timeframe, use weekly trend (via 8/21 EMA crossover) to filter RSI mean-reversion entries. 
Long when weekly uptrend + RSI < 30 + price below Bollinger lower band. 
Short when weekly downtrend + RSI > 70 + price above Bollinger upper band. 
Weekly trend filter avoids counter-trend trades; RSI extremes + Bollinger bands capture mean-reversion in both bull/bear markets. 
Designed for low trade frequency (<25/year) to minimize fee drag.
"""

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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 21:
        return np.zeros(n)
    
    # Calculate weekly 8 and 21 EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema8_weekly = pd.Series(close_weekly).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema8_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema8_weekly)
    ema21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema21_weekly)
    
    # Weekly trend: bullish when EMA8 > EMA21
    weekly_uptrend = ema8_weekly_aligned > ema21_weekly_aligned
    weekly_downtrend = ema8_weekly_aligned < ema21_weekly_aligned
    
    # Daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Bollinger Bands (20-period, 2 std)
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    lower_band = ma20 - (2 * std20)
    upper_band = ma20 + (2 * std20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Wait for weekly EMA21 and daily indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8_weekly_aligned[i]) or np.isnan(ema21_weekly_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(lower_band[i]) or np.isnan(upper_band[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: weekly trend alignment + RSI extreme + Bollinger band touch
        long_entry = weekly_uptrend[i] and (rsi[i] < 30) and (close[i] <= lower_band[i])
        short_entry = weekly_downtrend[i] and (rsi[i] > 70) and (close[i] >= upper_band[i])
        
        # Exit when RSI returns to neutral zone (40-60)
        long_exit = rsi[i] > 40
        short_exit = rsi[i] < 60
        
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

name = "1d_RSI_MeanRev_with_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0