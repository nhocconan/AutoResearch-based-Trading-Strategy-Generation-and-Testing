#!/usr/bin/env python3
"""
4h Bollinger Band Squeeze Breakout with 12h EMA Trend Filter
Long: Price breaks above upper BB after squeeze (BBW < 20th percentile) + price > 12h EMA34
Short: Price breaks below lower BB after squeeze + price < 12h EMA34
Exit: Opposite BB break or price crosses 12h EMA
Designed to capture volatility expansion after low volatility periods in both bull and bear markets.
Target: 80-150 total trades over 4 years (20-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # Bollinger Band Width for squeeze detection
    bbw = (upper - lower) / basis
    bbw_series = pd.Series(bbw)
    # 20th percentile lookback for squeeze definition
    bbw_percentile = bbw_series.rolling(window=50, min_periods=20).quantile(0.20)
    squeeze = bbw < bbw_percentile.values
    
    # Breakout detection
    breakout_up = (close > upper) & (close_series.shift(1) <= upper.shift(1))
    breakout_down = (close < lower) & (close_series.shift(1) >= lower.shift(1))
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need BB and EMA calculations
    
    for i in range(start_idx, n):
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: squeeze breakout up + price > 12h EMA34
            if squeeze[i] and breakout_up[i] and price > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down + price < 12h EMA34
            elif squeeze[i] and breakout_down[i] and price < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakout down OR price crosses below 12h EMA34
            if breakout_down[i] or price < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout up OR price crosses above 12h EMA34
            if breakout_up[i] or price > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BB_Squeeze_Breakout_12hEMA34"
timeframe = "4h"
leverage = 1.0