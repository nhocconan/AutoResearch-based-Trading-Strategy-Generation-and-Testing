#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with Daily Trend Filter
# - Williams %R(14) on 12h for overbought/oversold signals
# - Daily EMA(50) as trend filter: only long when price > EMA50, short when price < EMA50
# - Williams %R provides mean reversion signals in ranging markets
# - Daily EMA filter ensures alignment with higher timeframe trend
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on daily timeframe
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA to 12h timeframe
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 12h timeframe
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(ema_50_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        ema_filter = ema_50_12h[i]
        
        # Determine trend based on price relative to daily EMA50
        price_above_ema = price > ema_filter
        price_below_ema = price < ema_filter
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + price above daily EMA50
            if williams_r[i] < -80 and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + price below daily EMA50
            elif williams_r[i] > -20 and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) or price falls below EMA
            if williams_r[i] > -20 or price < ema_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) or price rises above EMA
            if williams_r[i] < -80 or price > ema_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_Filter"
timeframe = "12h"
leverage = 1.0