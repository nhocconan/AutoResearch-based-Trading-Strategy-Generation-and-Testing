#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA200 Trend Filter
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and Bear Power rising (less negative) with price above 1d EMA200
# Short when Bear Power < 0 and Bull Power falling (less positive) with price below 1d EMA200
# Uses 1d EMA200 for trend filter to avoid counter-trend trades
# Works in bull/bear by aligning with higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray components (13-period EMA)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: align with 1d EMA200
        uptrend = price > ema200_1d_aligned[i]
        downtrend = price < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power positive AND Bear Power rising (less negative) in uptrend
            if uptrend and bull_power[i] > 0 and bear_power[i] > bear_power[i-1]:
                position = 1
                signals[i] = position_size
            # Short: Bear Power negative AND Bull Power falling (less positive) in downtrend
            elif downtrend and bear_power[i] < 0 and bull_power[i] < bull_power[i-1]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bear Power turns positive (bulls losing control) or trend breaks
            if bear_power[i] >= 0 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bull Power turns negative (bears losing control) or trend breaks
            if bull_power[i] <= 0 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_1dEMA200_TrendFilter"
timeframe = "6h"
leverage = 1.0