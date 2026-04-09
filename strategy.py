#!/usr/bin/env python3
# 1d_kama_1w_trend_volume_v1
# Hypothesis: 1d strategy using Kaufman Adaptive Moving Average (KAMA) from 1w for trend direction, 
# with volume confirmation and discrete sizing (±0.25) to target 30-100 trades over 4 years.
# Long: price > 1w KAMA + volume spike
# Short: price < 1w KAMA + volume spike
# Exit: opposite signal or volume drop below average
# KAMA adapts to market noise, reducing whipsaw in ranging markets while capturing trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for KAMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Kaufman Adaptive Moving Average (KAMA) - 30 period, fast=2, slow=30
    # Based on Perry Kaufman's algorithm
    close_1w_series = pd.Series(close_1w)
    
    # Direction = absolute change over lookback period
    direction = abs(close_1w_series - close_1w_series.shift(10))  # 10-period change
    
    # Volatility = sum of absolute changes over lookback period
    volatility = abs(close_1w_series - close_1w_series.shift(1)).rolling(window=10, min_periods=10).sum()
    
    # Efficiency Ratio (ER) = Direction / Volatility
    er = direction / volatility.replace(0, np.nan)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Scaling factor (SC) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA = previous KAMA + SC * (price - previous KAMA)
    kama = np.full_like(close_1w, np.nan)
    kama[0] = close_1w[0]  # seed
    
    for i in range(1, len(close_1w)):
        if np.isnan(sc.iloc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below KAMA OR volume drops below average
            if close[i] < kama_aligned[i] or volume[i] < volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above KAMA OR volume drops below average
            if close[i] > kama_aligned[i] or volume[i] < volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price above KAMA
                if close[i] > kama_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price below KAMA
                elif close[i] < kama_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals