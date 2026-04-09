#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 1w trend filter
# Williams %R(14) identifies overbought/oversold conditions on daily timeframe
# Weekly EMA(21) determines primary trend direction for bias
# Entry: Williams %R < -80 (oversold) + price > weekly EMA(21) = long
# Entry: Williams %R > -20 (overbought) + price < weekly EMA(21) = short
# Exit: Williams %R returns to -50 level (mean reversion)
# Works in bull/bear: mean reversion in ranges, trend filter prevents counter-trend in strong moves
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "6h_1d_1w_williamsr_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
                          -50)  # neutral when range=0
    
    # Calculate 1w EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        williams_r_val = williams_r_aligned[i]
        ema_trend = ema_21_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit when Williams %R returns to -50 (mean reversion)
            if williams_r_val > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R returns to -50 (mean reversion)
            if williams_r_val < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: oversold + price above weekly EMA (bullish bias)
            if williams_r_val < -80 and close[i] > ema_trend:
                position = 1
                signals[i] = 0.25
            # Enter short: overbought + price below weekly EMA (bearish bias)
            elif williams_r_val > -20 and close[i] < ema_trend:
                position = -1
                signals[i] = -0.25
    
    return signals