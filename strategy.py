#!/usr/bin/env python3
"""
1h_4h_1d_Trend_Following
Hypothesis: Use 4h EMA50/200 crossover for trend direction and 1d EMA50/200 for regime filter, then enter on 1h pullbacks to EMA20 during 08-20 UTC session. This combines multi-timeframe trend alignment with mean-reversion entries to capture swings in both bull and bear markets while limiting trades via session filter and strict entry conditions.
"""

name = "1h_4h_1d_Trend_Following"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA50 and EMA200
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Determine trend: 1 if EMA50 > EMA200, -1 if EMA50 < EMA200
    trend_4h = np.where(ema50_4h > ema200_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50 and EMA200
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Determine regime: 1 if EMA50 > EMA200 (bull), -1 if EMA50 < EMA200 (bear)
    regime_1d = np.where(ema50_1d > ema200_1d, 1, -1)
    regime_1d_aligned = align_htf_to_ltf(prices, df_1d, regime_1d)
    
    # Calculate 1h EMA20 for pullback entries
    ema20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(regime_1d_aligned[i]) or 
            np.isnan(ema20_1h[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Only trade when 4h trend and 1d regime agree
        if trend_4h_aligned[i] == regime_1d_aligned[i]:
            direction = trend_4h_aligned[i]  # 1 for long, -1 for short
            
            if position == 0:
                # Enter long on pullback to EMA20 during uptrend
                if direction == 1 and low[i] <= ema20_1h[i]:
                    signals[i] = 0.20
                    position = 1
                # Enter short on pullback to EMA20 during downtrend
                elif direction == -1 and high[i] >= ema20_1h[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit long if trend breaks or price moves against us
                if trend_4h_aligned[i] != 1 or close[i] < ema20_1h[i] - 0.5 * (high[i] - low[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short if trend breaks or price moves against us
                if trend_4h_aligned[i] != -1 or close[i] > ema20_1h[i] + 0.5 * (high[i] - low[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # No trade when 4h trend and 1d regime disagree
            signals[i] = 0.0
    
    return signals