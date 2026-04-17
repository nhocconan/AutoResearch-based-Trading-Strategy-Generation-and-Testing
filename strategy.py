#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R with 1w EMA200 Trend Filter.
Long when Williams %R < -80 (oversold) AND price > 1w EMA200 (long-term uptrend).
Short when Williams %R > -20 (overbought) AND price < 1w EMA200 (long-term downtrend).
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR weekly trend reverses.
Uses 1d for Williams %R calculation, 1w for EMA200 trend filter.
Target: 30-100 total trades over 4 years (7-25/year). Williams %R captures mean reversion extremes,
weekly EMA200 filters for higher-timeframe trend alignment to reduce false signals in chop and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Williams %R on 1d timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1w EMA200 to 1d timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        price = close[i]
        ema200 = ema200_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1w EMA200 (long-term uptrend)
            if wr < -80 and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1w EMA200 (long-term downtrend)
            elif wr > -20 and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -50 (momentum weakening) OR price < 1w EMA200 (trend reversal)
            if wr > -50 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -50 (momentum weakening) OR price > 1w EMA200 (trend reversal)
            if wr < -50 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_WeeklyEMA200_Trend"
timeframe = "1d"
leverage = 1.0