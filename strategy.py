#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_Power_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (using daily data)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray components: Bull Power and Bear Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 6-period RSI for overbought/oversold conditions
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema13_val = ema13_1d_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, price above weekly EMA, RSI not overbought
            if (bull_power_val > 0 and 
                close[i] > ema34_1w_val and 
                rsi_val < 70):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, price below weekly EMA, RSI not oversold
            elif (bear_power_val < 0 and 
                  close[i] < ema34_1w_val and 
                  rsi_val > 30):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR price crosses below weekly EMA
            if (bull_power_val <= 0 or close[i] < ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR price crosses above weekly EMA
            if (bear_power_val >= 0 or close[i] > ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Uses Elder Ray (Bull/Bear Power) with weekly trend filter on 6-hour timeframe.
# - Elder Ray measures bull/bear power relative to 13-day EMA
# - Weekly EMA(34) filter ensures trading with higher timeframe trend
# - Long when Bull Power > 0, price above weekly EMA, RSI < 70
# - Short when Bear Power < 0, price below weekly EMA, RSI > 30
# - Exits when power shifts or price crosses weekly EMA
# - Works in both bull and bear markets by adapting to weekly trend direction
# - Position size: 0.25 for balanced risk/return
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Focus on BTC and ETH as primary targets (not SOL-only)