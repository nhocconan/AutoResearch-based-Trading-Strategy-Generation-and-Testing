#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance zones.
A breakout above resistance or below support with volume expansion indicates institutional participation
and often leads to sustained moves. This strategy works in both bull (breakouts to upside) and bear
(breakdowns to downside) markets by trading the direction of the breakout.
Uses volume confirmation and a volatility filter to avoid false breakouts in low-volatility environments.
Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    # Resistance levels
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    R2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    # Support levels
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    S2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    S4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align each level to lower timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Volatility filter: avoid trading in extremely low volatility
    # Use 20-period ATR percentage
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean()
    atr_percent = atr_20 / close * 100
    # Only trade when volatility is above 20th percentile (avoid choppy low-vol periods)
    vol_threshold = pd.Series(atr_percent).rolling(window=50, min_periods=20).quantile(0.2)
    volatility_filter = atr_percent > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(volume_expansion[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R3 with volume expansion
        long_breakout = close[i] > R3_aligned[i] and volume_expansion[i] and volatility_filter[i]
        
        # Short breakdown: price breaks below S3 with volume expansion
        short_breakdown = close[i] < S3_aligned[i] and volume_expansion[i] and volatility_filter[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakdown and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0