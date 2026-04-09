#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d Williams %R regime filter
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# - Williams %R on 1d identifies overbought/oversold conditions
# - Long when Bull Power > 0 AND Williams %R < -80 (oversold in downtrend = bounce)
# - Short when Bear Power > 0 AND Williams %R > -20 (overbought in uptrend = pullback)
# - Uses EMA13 for trend reference (fast enough for 6h, smooth enough for noise)
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies)
# - Williams %R regime filter prevents trading against strong momentum
# - Discrete position size: 0.25 to minimize fee churn

name = "6h_1d_elderray_williamsr_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Williams %R
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r_1d = np.where(highest_high_14 == lowest_low_14, -50, williams_r_1d)
    
    # Align 1d Williams %R to 6h
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(williams_r_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Bull Power turns negative OR Williams %R becomes overbought
            if bull_power[i] <= 0 or williams_r_1d_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Bear Power turns negative OR Williams %R becomes oversold
            if bear_power[i] <= 0 or williams_r_1d_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entry conditions
            # Long: Bull Power positive AND Williams %R oversold (< -80)
            # Short: Bear Power positive AND Williams %R overbought (> -20)
            if bull_power[i] > 0 and williams_r_1d_aligned[i] < -80:
                position = 1
                signals[i] = 0.25
            elif bear_power[i] > 0 and williams_r_1d_aligned[i] > -20:
                position = -1
                signals[i] = -0.25
    
    return signals