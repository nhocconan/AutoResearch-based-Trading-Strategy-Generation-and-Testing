#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA trend filter with daily Bollinger Band mean reversion entries.
# Uses Kaufman's Adaptive Moving Average (KAMA) to identify trend direction on 12h,
# and enters mean-reversion trades when price touches Bollinger Bands (20,2) on daily,
# only when the 12h trend aligns with the mean reversion direction.
# Trend filter reduces whipsaws, Bollinger Bands provide clear entry/exit levels.
# Designed for low trade frequency (12-37/year) to minimize fee drag in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h KAMA (trend filter)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate daily Bollinger Bands (20,2)
    close_1d = df_1d['close'].values
    basis = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    dev = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # Align all data to 12-hour timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA is already 12h, but align for safety
    basis_aligned = align_htf_to_ltf(prices, df_1d, basis)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(basis_aligned[i]) or
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition: price above/below KAMA
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # Bollinger Band conditions
        price_at_upper = close[i] >= upper_aligned[i]
        price_at_lower = close[i] <= lower_aligned[i]
        
        # Entry logic: mean reversion in direction of trend
        if position == 0:
            # Long when price touches lower BB and trend is up (price > KAMA)
            if price_at_lower and price_above_kama:
                position = 1
                signals[i] = position_size
            # Short when price touches upper BB and trend is down (price < KAMA)
            elif price_at_upper and price_below_kama:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price returns to basis (mean reversion complete) or trend fails
            if close[i] >= basis_aligned[i] or close[i] <= kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price returns to basis or trend fails
            if close[i] <= basis_aligned[i] or close[i] >= kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_BB_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0