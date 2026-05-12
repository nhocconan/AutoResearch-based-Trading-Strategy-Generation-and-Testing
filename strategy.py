#!/usr/bin/env python3
"""
6h_Weekly_Pivot_AntiTrend_Fade
Hypothesis: In crypto markets, price often reverses at weekly pivot extremes (R4/S4) during consolidation phases, 
especially when combined with overbought/oversold conditions on the 6h timeframe. This strategy fades extreme 
weekly pivot touches using RSI extremes as confirmation, aiming to capture mean reversion swings in both bull 
and bear markets. Weekly pivots provide structure that adapts to changing volatility, while RSI prevents 
entering during strong momentum. Designed for low trade frequency to avoid fee drag on 6h timeframe.
"""

name = "6h_Weekly_Pivot_AntiTrend_Fade"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(14) for overbought/oversold conditions
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard floor trader pivots)
    pivot = (high_1w + low_1w + close_1w) / 3
    r4 = pivot + 3 * (high_1w - low_1w)  # R4 = P + 3*(H-L)
    s4 = pivot - 3 * (high_1w - low_1w)  # S4 = P - 3*(H-L)
    
    # Align weekly pivot levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches or goes below S4 (weekly support) AND RSI oversold (<30)
            if low[i] <= s4_aligned[i] and rsi_values[i] < 30:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or goes above R4 (weekly resistance) AND RSI overbought (>70)
            elif high[i] >= r4_aligned[i] and rsi_values[i] > 70:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot OR RSI reaches neutral (50)
            if high[i] >= pivot[i] or rsi_values[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot OR RSI reaches neutral (50)
            if low[i] <= pivot[i] or rsi_values[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals