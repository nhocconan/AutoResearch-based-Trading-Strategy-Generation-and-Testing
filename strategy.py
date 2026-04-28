#!/usr/bin/env python3
"""
4h_Stochastic_Trend_Squeeze
Hypothesis: Combine stochastic oscillator with trend filter and volatility squeeze on 4h timeframe.
Long when stochastic > 80 in uptrend with volatility contraction; short when stochastic < 20 in downtrend with volatility contraction.
Uses weekly trend filter to avoid counter-trend trades. Targets 25-35 trades/year to minimize fee drag while capturing momentum extremes.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Stochastic oscillator (14,3,3) on 4h
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean()
    
    # Volatility squeeze: Bollinger Band width < 20th percentile
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True)
    squeeze = bb_width_percentile < 0.2  # Bottom 20% = volatility contraction
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(k_percent[i]) or np.isnan(d_percent[i]) or
            np.isnan(squeeze[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Stochastic conditions
        stoch_overbought = k_percent[i] > 80 and d_percent[i] > 80
        stoch_oversold = k_percent[i] < 20 and d_percent[i] < 20
        stoch_cross_up = k_percent[i-1] <= d_percent[i-1] and k_percent[i] > d_percent[i]
        stoch_cross_down = k_percent[i-1] >= d_percent[i-1] and k_percent[i] < d_percent[i]
        
        # Entry logic: stochastic extreme in trend direction with volatility squeeze
        long_entry = squeeze[i] and weekly_uptrend and stoch_oversold and stoch_cross_up
        short_entry = squeeze[i] and weekly_downtrend and stoch_overbought and stoch_cross_down
        
        # Exit logic: opposite stochastic cross or trend change
        long_exit = stoch_cross_down or (not weekly_uptrend)
        short_exit = stoch_cross_up or (not weekly_downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Stochastic_Trend_Squeeze"
timeframe = "4h"
leverage = 1.0