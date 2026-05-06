#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1-day trend filter
# Williams %R(14) measures overbought/oversold levels. Buy when %R crosses above -80 from oversold
# in a 1-day uptrend (close > EMA50). Sell when %R crosses below -20 from overbought
# in a 1-day downtrend (close < EMA50). Uses mean reversion in ranging markets
# while trend filter prevents counter-trend trades. Effective in both bull/bear regimes
# as it buys dips in uptrends and sells rallies in downtrends. Target: 60-120 trades over 4 years.

name = "6h_WilliamsR_MeanReversion_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1-day Williams %R and EMA50 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1-day indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r_aligned[i]
        ema50 = ema_50_aligned[i]
        
        if position == 0:
            # Long signal: Williams %R crosses above -80 from oversold in uptrend
            if wr > -80 and wr <= -79 and close[i] > ema50:
                signals[i] = 0.25
                position = 1
            # Short signal: Williams %R crosses below -20 from overbought in downtrend
            elif wr < -20 and wr >= -21 and close[i] < ema50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) or trend turns bearish
            if wr >= -20 or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) or trend turns bullish
            if wr <= -80 or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals