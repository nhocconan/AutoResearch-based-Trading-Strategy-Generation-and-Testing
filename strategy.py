#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1w EMA50 trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA50 trend filter, 1d for Elder Ray calculation.
- Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend direction and avoids ranging markets.
- Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures trend strength.
- Long when: Alligator aligned bullish (lips > teeth > jaw) AND Bull Power > 0 AND price > 1w EMA50.
- Short when: Alligator aligned bearish (lips < teeth < jaw) AND Bear Power < 0 AND price < 1w EMA50.
- Exit: Opposite Alligator alignment OR price crosses 1w EMA50.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets (buy during Alligator uptrend) and bear markets (sell during Alligator downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Alligator trend persistence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    ema13_1d = ema(df_1d['close'].values, 13)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d, additional_delay_bars=1)
    
    # Williams Alligator on 6h (jaw=13, teeth=8, lips=5)
    jaw = sma(close, 13)  # Slowest (13-period)
    teeth = sma(close, 8)  # Medium (8-period)
    lips = sma(close, 5)   # Fastest (5-period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema13_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Exit conditions: opposite Alligator alignment OR price crosses 1w EMA50
        if position != 0:
            # Exit long: Alligator turns bearish OR price falls below 1w EMA50
            if position == 1:
                if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Alligator turns bullish OR price rises above 1w EMA50
            elif position == -1:
                if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with Elder Ray confirmation and trend filter
        if position == 0:
            # Bullish Alligator: lips > teeth > jaw
            bullish_alligator = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator: lips < teeth < jaw
            bearish_alligator = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
            bull_power = curr_high - ema13_1d_aligned[i]
            bear_power = curr_low - ema13_1d_aligned[i]
            
            # Long: Bullish Alligator AND Bull Power > 0 AND price > 1w EMA50
            if bullish_alligator and bull_power > 0 and curr_close > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND Bear Power < 0 AND price < 1w EMA50
            elif bearish_alligator and bear_power < 0 and curr_close < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Alligator_ElderRay_1wEMA50_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0