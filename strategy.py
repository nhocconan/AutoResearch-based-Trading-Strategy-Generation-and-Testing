#!/usr/bin/env python3
"""
6h Bollinger Band Squeeze Breakout with 1d Trend Filter
Long: Bollinger Band width at 6-month low + price breaks above upper band + 1d EMA50 rising
Short: Bollinger Band width at 6-month low + price breaks below lower band + 1d EMA50 falling
Exit: Opposite band break or price crosses middle band
Targets volatility contraction/expansion cycles, works in both trends and ranges.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # need enough data for 20-period BB and 50-period EMA
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Bollinger Bands on 6h (20-period, 2 std dev)
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    bb_width = upper - lower
    
    # 6-month low of BB width (approx 180 days * 4 = 720 bars of 6h data)
    # Use 100-period lookback for 6-month low approximation
    bb_width_low = pd.Series(bb_width).rolling(window=100, min_periods=100).min()
    squeeze_condition = bb_width <= bb_width_low * 1.1  # within 10% of lowest width
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_slope = np.diff(ema50_1d_aligned, prepend=ema50_1d_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need BB width lookback
    
    for i in range(start_idx, n):
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(bb_width[i]) or np.isnan(bb_width_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_slope[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        middle = basis[i]
        
        if position == 0:
            # Long: squeeze + break above upper + rising 1d EMA50
            if squeeze_condition[i] and price > upper[i] and ema50_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: squeeze + break below lower + falling 1d EMA50
            elif squeeze_condition[i] and price < lower[i] and ema50_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower band OR crosses below middle
            if price < lower[i] or price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper band OR crosses above middle
            if price > upper[i] or price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_Breakout_Trend"
timeframe = "6h"
leverage = 1.0