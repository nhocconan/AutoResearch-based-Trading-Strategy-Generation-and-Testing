#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w trend filter
# Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R < -80 (oversold) and weekly trend is up
# Short when %R > -20 (overbought) and weekly trend is down
# Uses 14-period lookback, targets 10-20 trades/year
# Williams %R is mean-reversion oscillator that works in ranging markets
# Weekly trend filter ensures we trade with the higher timeframe momentum
# Works in bull/bear: in uptrend, buy oversold; in downtrend, sell overbought

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 14-period Williams %R
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, 
                          (highest_high - close_1d) / denominator * -100, 
                          -50.0)  # neutral when no range
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly trend: 5-period EMA
    ema_5 = pd.Series(close_1w).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align indicators to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_5_aligned = align_htf_to_ltf(prices, df_1w, ema_5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_5_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        weekly_trend_up = ema_5_aligned[i] > 0  # EMA value itself indicates trend
        
        # Long: Williams %R oversold (< -80) and weekly trend up
        if wr < -80 and weekly_trend_up:
            signals[i] = 0.25
            position = 1
        # Short: Williams %R overbought (> -20) and weekly trend down
        elif wr > -20 and not weekly_trend_up:
            signals[i] = -0.25
            position = -1
        else:
            # Hold position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WilliamsR_1wTrend"
timeframe = "1d"
leverage = 1.0