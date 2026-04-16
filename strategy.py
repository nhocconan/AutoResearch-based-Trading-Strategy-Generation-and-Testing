#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Williams %R extremes with 1d EMA50 trend filter.
# Long when weekly %R < -80 (oversold) AND close > 1d EMA50 (uptrend).
# Short when weekly %R > -20 (overbought) AND close < 1d EMA50 (downtrend).
# Exit when %R returns to normal range (-50) or price crosses EMA50.
# Uses discrete position size 0.25. Weekly %R captures extreme sentiment reversals.
# 1d EMA50 ensures trading with intermediate trend to avoid counter-trend whipsaws.
# 6h timeframe targets 80-160 total trades over 4 years (20-40/year) for optimal fee balance.
# Works in bull markets (buy oversold dips in uptrends) and bear markets (sell overbought rallies in downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data once before loop for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data once before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === Weekly Indicators: Williams %R (14-period) ===
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = (highest_high_14 - close_1w) / (highest_high_14 - lowest_low_14) * -100
    williams_r_1w = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_1w)
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        williams_r = williams_r_aligned[i]
        ema50 = ema50_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when %R > -50 (leaving oversold) OR price < EMA50 (trend break)
            if (williams_r > -50) or (price < ema50):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when %R < -50 (leaving overbought) OR price > EMA50 (trend break)
            if (williams_r < -50) or (price > ema50):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: %R < -80 (extreme oversold) AND price > EMA50 (uptrend)
            if (williams_r < -80) and (price > ema50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: %R > -20 (extreme overbought) AND price < EMA50 (downtrend)
            elif (williams_r > -20) and (price < ema50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1wWilliamsR_Extreme_1dEMA50_TrendFilter_V1"
timeframe = "6h"
leverage = 1.0