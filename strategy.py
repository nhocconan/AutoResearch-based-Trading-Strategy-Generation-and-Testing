#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme Reversion with Weekly Trend Filter.
Long when Williams %R < -80 (oversold) AND price > weekly EMA200 (bullish bias).
Short when Williams %R > -20 (overbought) AND price < weekly EMA200 (bearish bias).
Exit when Williams %R reverts to -50 (mean reversion) OR weekly trend reverses.
Uses 1w for EMA200 trend filter, 12h for Williams %R calculation.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R captures extreme momentum,
weekly EMA200 filters for higher-timeframe trend alignment to reduce false signals in chop.
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
    
    # Calculate Williams %R on 12h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Align 1w EMA200 to 12h timeframe
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
            # Long: Williams %R < -80 (oversold) AND price > weekly EMA200 (bullish bias)
            if wr < -80 and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < weekly EMA200 (bearish bias)
            elif wr > -20 and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R >= -50 (reverted from oversold) OR price < weekly EMA200 (trend reversal)
            if wr >= -50 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R <= -50 (reverted from overbought) OR price > weekly EMA200 (trend reversal)
            if wr <= -50 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_WeeklyEMA200_Trend"
timeframe = "12h"
leverage = 1.0