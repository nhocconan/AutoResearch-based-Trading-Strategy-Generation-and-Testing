#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with weekly trend filter for mean reversion.
# Williams %R identifies overbought/oversold conditions.
# Weekly trend filter ensures we trade in direction of higher timeframe trend.
# Target: 20-40 trades/year per symbol with good risk/reward.
name = "1d_WilliamsR_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def williams_r(high, low, close, period=14):
    """Williams %R indicator"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA50 for trend direction
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Calculate Williams %R on daily
    wr = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure weekly EMA and WR are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_50_weekly_aligned[i]) or np.isnan(wr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_weekly_val = ema_50_weekly_aligned[i]
        wr_value = wr[i]
        
        # Weekly trend filter
        uptrend = price > ema_50_weekly_val
        downtrend = price < ema_50_weekly_val
        
        if position == 0:
            # Enter long when oversold in uptrend
            if wr_value < -80 and uptrend:
                signals[i] = 0.25
                position = 1
            # Enter short when overbought in downtrend
            elif wr_value > -20 and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when overbought or trend changes
            if wr_value > -20 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when oversold or trend changes
            if wr_value < -80 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals