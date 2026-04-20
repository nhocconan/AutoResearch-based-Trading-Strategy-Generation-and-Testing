#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + 1w Trend Filter
# - Williams %R (14) on 1d for mean reversion signals
# - Long when %R < -80 (oversold) and 1w EMA21 > 1w EMA50 (bullish trend)
# - Short when %R > -20 (overbought) and 1w EMA21 < 1w EMA50 (bearish trend)
# - Uses weekly EMA crossover for trend filter to avoid counter-trend trades
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d timeframe
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_1d = williams_r.values
    
    # Align 1d Williams %R to 1d timeframe (no conversion needed)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Load 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA21 and EMA50 on 1w timeframe
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA values to 1d timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if NaN in indicators
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r = williams_r_1d_aligned[i]
        ema21 = ema21_1w_aligned[i]
        ema50 = ema50_1w_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + bullish trend (EMA21 > EMA50)
            if williams_r < -80 and ema21 > ema50:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + bearish trend (EMA21 < EMA50)
            elif williams_r > -20 and ema21 < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns above -50 or trend turns bearish
            if williams_r > -50 or ema21 < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns below -50 or trend turns bullish
            if williams_r < -50 or ema21 > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_1wEMA_TrendFilter"
timeframe = "1d"
leverage = 1.0