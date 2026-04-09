#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1w trend filter and volume confirmation
# In bull markets (price > 1w EMA50): look for Williams %R < -80 (oversold) for long entries
# In bear markets (price < 1w EMA50): look for Williams %R > -20 (overbought) for short entries
# Volume confirmation requires 2x average volume to avoid false signals
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in both bull and bear markets by adapting to 1w trend direction

name = "6h_1w_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    close_1w_s = pd.Series(close_1w)
    ema50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0,
                          -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14),
                          -50)
    
    # Calculate 6h average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 2.0 * avg_volume
    
    # Align 1w indicators to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1w trend
        bull_trend = close[i] > ema50_1w_aligned[i]
        bear_trend = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exiting oversold) or trend changes to bear
            if williams_r[i] > -50 or bear_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 (exiting overbought) or trend changes to bull
            if williams_r[i] < -50 or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long in bull trend when Williams %R < -80 (oversold) with volume confirmation
            if bull_trend and williams_r[i] < -80 and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            # Enter short in bear trend when Williams %R > -20 (overbought) with volume confirmation
            elif bear_trend and williams_r[i] > -20 and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals