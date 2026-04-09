#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Williams %R and 1d EMA filter for trend direction
# - Weekly Williams %R(14) identifies overbought/oversold conditions on completed weekly candles
# - Daily EMA(50) determines trend bias: price > EMA50 = bullish bias, price < EMA50 = bearish bias
# - Long when Williams %R < -80 (oversold) AND price > daily EMA50 (bullish alignment)
# - Short when Williams %R > -20 (overbought) AND price < daily EMA50 (bearish alignment)
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: mean reversion within trend filters avoid counter-trend traps
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_1w_1d_williamsr_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams %R(14) on 1w data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1w) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (wait for completed 1w bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d data
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA(50) to 6h timeframe (wait for completed 1d bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        williams_r_val = williams_r_aligned[i]
        ema_50_val = ema_50_aligned[i]
        price = close[i]
        
        # Long signal: oversold AND bullish trend alignment
        if williams_r_val < -80 and price > ema_50_val:
            signals[i] = 0.25
        # Short signal: overbought AND bearish trend alignment
        elif williams_r_val > -20 and price < ema_50_val:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals