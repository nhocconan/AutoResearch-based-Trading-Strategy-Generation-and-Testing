#!/usr/bin/env python3
# 6h_1w_Keltner_Channel_Breakout_Trend
# Hypothesis: On 6h timeframe, trade breakouts from 1-week Keltner Channel with trend filter from 1-day EMA.
# Uses weekly ATR-based Keltner Channel (20, 1.5) to identify volatility breakouts.
# Only takes long when price > weekly upper band and 6h price > daily EMA50 (uptrend).
# Only takes short when price < weekly lower band and 6h price < daily EMA50 (downtrend).
# Weekly channel ensures we trade only significant breakouts, daily EMA filters for trend alignment.
# Targets 15-30 trades per year. Works in bull/bear via trend-aligned breakouts.
# Volatility breakouts capture momentum after consolidation; trend filter avoids counter-trend whipsaws.

name = "6h_1w_Keltner_Channel_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week Keltner Channel (20, 1.5)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for EMA base
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    
    # EMA of typical price (20-period)
    typical_ema_20 = pd.Series(typical_price_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR (10-period) for channel width
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]  # first TR
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Bands
    upper_1w = typical_ema_20 + 1.5 * atr_10
    lower_1w = typical_ema_20 - 1.5 * atr_10
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly Keltner bands to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Align daily EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly upper band and 6h price above daily EMA50 (uptrend)
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly lower band and 6h price below daily EMA50 (downtrend)
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below weekly lower band or below daily EMA50
            if close[i] < lower_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above weekly upper band or above daily EMA50
            if close[i] > upper_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals