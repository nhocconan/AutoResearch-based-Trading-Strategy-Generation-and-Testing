#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly Trend Filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
# Weekly trend filter ensures we only trade in direction of weekly trend (using weekly EMA40).
# Long when Bull Power > 0 and close > weekly EMA40.
# Short when Bear Power > 0 and close < weekly EMA40.
# Works in bull markets (captures strength) and bear markets (captures weakness).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # >0 indicates bullish strength
    bear_power = ema13 - low   # >0 indicates bearish strength
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 40:
        return np.zeros(n)
    weekly_close = df_weekly['close'].values
    
    # Weekly EMA40 for trend filter
    weekly_close_s = pd.Series(weekly_close)
    weekly_ema40 = weekly_close_s.ewm(span=40, adjust=False, min_periods=40).mean().values
    weekly_ema40_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema40)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if weekly trend data not available
        if np.isnan(weekly_ema40_aligned[i]):
            continue
        
        # Long entry: Bull Power positive AND price above weekly EMA40 (uptrend)
        if bull_power[i] > 0 and close[i] > weekly_ema40_aligned[i] and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short entry: Bear Power positive AND price below weekly EMA40 (downtrend)
        elif bear_power[i] > 0 and close[i] < weekly_ema40_aligned[i] and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite Elder Ray signal or trend change
        elif position == 1 and (bear_power[i] > 0 or close[i] < weekly_ema40_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bull_power[i] > 0 or close[i] > weekly_ema40_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_WeeklyTrend"
timeframe = "6h"
leverage = 1.0