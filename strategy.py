#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R regime filter
# - Uses 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) to measure bull/bear strength
# - Filters with 1d Williams %R: only take longs when %R < -80 (oversold) and shorts when %R > -20 (overbought)
# - Requires Elder Ray and Williams %R to agree on direction (confluence filter)
# - Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies)
# - Elder Ray identifies power shifts, Williams %R identifies extreme points for mean reversion

name = "6h_1d_elderray_williamsr_v1"
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
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Williams %R(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14))
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # 1d Williams %R conditions: oversold < -80, overbought > -20
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    # Align 1d Williams %R to 6h
    williams_oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_oversold.astype(float))
    williams_overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_overbought.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 6h Elder Ray components
    bull_power = high - ema13  # measures bull strength
    bear_power = ema13 - low   # measures bear strength
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(williams_oversold_aligned[i]) or np.isnan(williams_overbought_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when bear power exceeds bull power (momentum shift) OR Williams %R no longer oversold
            if bear_power[i] > bull_power[i] or not williams_oversold_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when bull power exceeds bear power (momentum shift) OR Williams %R no longer overbought
            if bull_power[i] > bear_power[i] or not williams_overbought_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for confluence: Elder Ray direction agrees with Williams %R extreme
            if bull_power[i] > bear_power[i] and williams_oversold_aligned[i]:
                # Bull power > bear power AND Williams %R oversold = long setup
                position = 1
                signals[i] = 0.25
            elif bear_power[i] > bull_power[i] and williams_overbought_aligned[i]:
                # Bear power > bull power AND Williams %R overbought = short setup
                position = -1
                signals[i] = -0.25
    
    return signals