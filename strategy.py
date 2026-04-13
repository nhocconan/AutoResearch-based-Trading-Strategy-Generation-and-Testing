#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter.
# Williams %R identifies overbought/oversold conditions for mean reversion.
# 1w EMA filter ensures trades align with higher timeframe trend.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period)
    willr_period = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(willr_period-1, n):
        highest_high[i] = np.max(high[i-willr_period+1:i+1])
        lowest_low[i] = np.min(low[i-willr_period+1:i+1])
    
    willr = np.full(n, np.nan)
    for i in range(willr_period-1, n):
        if highest_high[i] != lowest_low[i]:
            willr[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            willr[i] = -50  # neutral when no range
    
    # Calculate 20-period EMA on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros(len(close_1w))
    alpha = 2.0 / (20 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_1w[i] = close_1w[i]
        else:
            ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(willr[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        willr_val = willr[i]
        ema_1w_val = ema_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above weekly EMA
            if (willr_val < -80 and 
                price > ema_1w_val):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) + price below weekly EMA
            elif (willr_val > -20 and 
                  price < ema_1w_val):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or trend change
            if (willr_val > -50 or 
                price < ema_1w_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or trend change
            if (willr_val < -50 or 
                price > ema_1w_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_WilliamsR_MeanReversion_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0