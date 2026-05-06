#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme reversal with 1d EMA50 trend filter
# Long when 1d Williams %R crosses above -80 (oversold) AND 1d EMA50 > EMA200 AND close > EMA50
# Short when 1d Williams %R crosses below -20 (overbought) AND 1d EMA50 < EMA200 AND close < EMA50
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme
# Uses discrete sizing 0.25 to control drawdown and fees
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R identifies exhaustion points; EMA50/EMA200 filters ensure trend alignment
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)

name = "4h_1dWilliamsR_ExtremeReversal_1dEMA50Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for Williams %R and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero (when high == low)
    williams_r_1d = np.where(highest_high_14 == lowest_low_14, -50, williams_r_1d)
    
    # Align 1d Williams %R to 4h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA values to 4h timeframe (wait for completed 1d bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold recovery) in uptrend
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                ema_50_aligned[i] > ema_200_aligned[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought decline) in downtrend
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  ema_50_aligned[i] < ema_200_aligned[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) or crosses below -80 (re-oversold)
            if williams_r_aligned[i] <= -50 or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) or crosses above -20 (re-overbought)
            if williams_r_aligned[i] >= -50 or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals