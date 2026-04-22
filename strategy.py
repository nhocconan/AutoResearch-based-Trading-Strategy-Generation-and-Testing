#!/usr/bin/env python3

"""
Hypothesis: Daily Bollinger Band Width Breakout with Weekly Trend Filter.
Trades breakouts of Bollinger Bands when weekly EMA trend is established and Bollinger Band Width is low (squeeze).
Uses volatility contraction followed by expansion to capture trend initiation in both bull and bear markets.
Designed for very low trade frequency (<10 trades/year) to minimize fee drag and work in all market regimes.
"""

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def calculate_bollinger_bands(close, length=20, mult=2.0):
    """Calculate Bollinger Bands: middle, upper, lower."""
    basis = pd.Series(close).rolling(window=length, min_periods=length).mean().values
    dev = pd.Series(close).rolling(window=length, min_periods=length).std().values
    upper = basis + mult * dev
    lower = basis - mult * dev
    return basis, upper, lower

def calculate_bb_width(upper, lower):
    """Calculate Bollinger Band Width: (upper - lower) / middle."""
    # Avoid division by zero
    return (upper - lower) / np.where(upper + lower == 0, 1e-10, upper + lower) * 2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Bollinger Bands (20, 2.0)
    basis, upper, lower = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(upper, lower)
    
    # Bollinger Band Width percentile (20-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bollinger Band Squeeze condition: width in lowest 20% percentile
        squeeze = bb_width_percentile[i] <= 0.2
        
        if position == 0 and squeeze:
            # Long: price breaks above upper band with weekly uptrend
            if close[i] > upper[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower band with weekly downtrend
            elif close[i] < lower[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: price returns to middle band (mean reversion) or volatility expands too much
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle band or BB width expands significantly
                if close[i] < basis[i] or bb_width_percentile[i] >= 0.8:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle band or BB width expands significantly
                if close[i] > basis[i] or bb_width_percentile[i] >= 0.8:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "Daily_Bollinger_Width_Squeeze_WeeklyEMA34_Trend"
timeframe = "1d"
leverage = 1.0