#!/usr/bin/env python3
"""
Hypothesis: 6-hour Bollinger Band squeeze with 1-week trend filter.
Long when price breaks above upper band during low volatility (BB width < 20th percentile) and 1-week EMA200 rising.
Short when price breaks below lower band during low volatility and 1-week EMA200 falling.
Exit when price returns to middle band or volatility expands (BB width > 80th percentile).
This strategy captures breakouts from low volatility regimes, which often precede strong moves.
The 1-week EMA200 filter ensures alignment with higher timeframe trend, improving performance in both bull and bear markets.
By requiring both volatility contraction and expansion, trade frequency is naturally limited.
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
    
    # Load 1-week data for EMA200 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Bollinger Bands (20, 2)
    bb_window = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    std = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    upper = sma + (std * bb_std)
    lower = sma - (std * bb_std)
    bb_width = upper - lower
    
    # Percentiles for volatility regime (using expanding window to avoid look-ahead)
    bb_width_pct = np.full_like(bb_width, np.nan)
    for i in range(bb_window, n):
        # Use historical data up to i-1 to compute percentile
        historical_width = bb_width[bb_window:i]  # Exclude current to avoid look-ahead
        if len(historical_width) >= 20:  # Need minimum samples
            current_width = bb_width[i]
            percentile = np.sum(historical_width <= current_width) / len(historical_width) * 100
            bb_width_pct[i] = percentile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_window, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(bb_width_pct[i]) or np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Low volatility breakout conditions
            low_vol = bb_width_pct[i] < 20  # Bottom 20% of volatility
            high_vol = bb_width_pct[i] > 80  # Top 20% of volatility
            
            # Long: Price breaks above upper band during low vol and weekly uptrend
            if (low_vol and 
                close[i] > upper[i] and 
                ema200_1w_aligned[i] > ema200_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band during low vol and weekly downtrend
            elif (low_vol and 
                  close[i] < lower[i] and 
                  ema200_1w_aligned[i] < ema200_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle band OR volatility expands significantly
                if (close[i] < sma[i] or 
                    bb_width_pct[i] > 80):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle band OR volatility expands significantly
                if (close[i] > sma[i] or 
                    bb_width_pct[i] > 80):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Bollinger_Squeeze_1wEMA200_Trend"
timeframe = "6h"
leverage = 1.0