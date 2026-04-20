#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA Trend Filter
# - Williams %R(14) on 4h for overbought/oversold signals
# - Long when %R < -80 (oversold) and 1d EMA(50) > 1d EMA(200) (bullish trend)
# - Short when %R > -20 (overbought) and 1d EMA(50) < 1d EMA(200) (bearish trend)
# - Uses EMA crossover on daily timeframe to filter for intermediate trend direction
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) and EMA(200) on 1d timeframe
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Determine trend: 1 = bullish (EMA50 > EMA200), -1 = bearish (EMA50 < EMA200)
    trend_1d = np.where(ema_50_1d > ema_200_1d, 1, -1)
    
    # Align 1d trend to 4h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate Williams %R (14) on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_4h) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Williams %R warmup
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(trend_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        wr = williams_r[i]
        trend = trend_1d_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + bullish 1d trend
            if wr < -80 and trend == 1:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + bearish 1d trend
            elif wr > -20 and trend == -1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 or trend turns bearish
            if wr > -50 or trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 or trend turns bullish
            if wr < -50 or trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA_TrendFilter"
timeframe = "4h"
leverage = 1.0