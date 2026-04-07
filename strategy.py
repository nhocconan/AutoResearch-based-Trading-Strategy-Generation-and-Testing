#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R(14) mean reversion with 12-hour trend filter
# Long when Williams %R < -80 (oversold) in 12h uptrend
# Short when Williams %R > -20 (overbought) in 12h downtrend
# Exit when Williams %R crosses above -50 (long) or below -50 (short)
# Uses tight entries to target 50-150 total trades over 4 years (12-37/year)
# Williams %R identifies exhaustion points; trend filter avoids counter-trend trades
# Effective in both bull (buy oversold dips) and bear (sell overbought rallies) markets

name = "6h_williamsr_meanrev_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(25) and EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R(14) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema_25_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(willr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Williams %R crosses above -50 (exhaustion fading)
            if willr[i] > -50 and willr[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses below -50 (exhaustion fading)
            if willr[i] < -50 and willr[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Trend filter: 12h EMA(25) > EMA(50) for uptrend, < for downtrend
            uptrend = ema_25_12h_aligned[i] > ema_50_12h_aligned[i]
            downtrend = ema_25_12h_aligned[i] < ema_50_12h_aligned[i]
            
            # Long: oversold in uptrend
            if willr[i] < -80 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: overbought in downtrend
            elif willr[i] > -20 and downtrend:
                signals[i] = -0.25
                position = -1
    
    return signals