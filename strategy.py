#!/usr/bin/env python3
"""
6h_EMA34_Trend_With_12h_Resistance_Support
Hypothesis: On 6h timeframe, use EMA(34) for trend direction and 12h support/resistance levels for entry timing.
Long when price pulls back to EMA(34) during uptrend (12h EMA34 > 12h EMA89) with rejection from 12h support.
Short when price rallies to EMA(34) during downtrend (12h EMA34 < 12h EMA89) with rejection from 12h resistance.
Uses rejection candles (pin bar patterns) to avoid false breakouts.
Works in both bull (buy dips) and bear (sell rallies) markets by following 12h trend.
Target: 20-40 trades/year by requiring trend alignment + pullback + rejection confirmation.
"""

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
    
    # Get 12h data for trend and S/R levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) and EMA(89) for trend
    if len(close_12h) >= 89:
        ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema89_12h = pd.Series(close_12h).ewm(span=89, adjust=False, min_periods=89).mean().values
    else:
        ema34_12h = np.full_like(close_12h, np.nan)
        ema89_12h = np.full_like(close_12h, np.nan)
    
    # Calculate 6h EMA(34) for dynamic support/resistance
    if len(close) >= 34:
        ema34_6h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema34_6h = np.full_like(close, np.nan)
    
    # Align 12h EMAs to 6h timeframe
    ema34_12h_6h = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema89_12h_6h = align_htf_to_ltf(prices, df_12h, ema89_12h)
    
    # Identify rejection candles (pin bars) on 6h
    # Bullish rejection: long lower wick, close near high
    body_size = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    
    open_ = prices['open'].values
    is_bullish_rejection = (lower_wick > 2 * body_size) & (body_size > 0) & (close > open_)
    is_bearish_rejection = (upper_wick > 2 * body_size) & (body_size > 0) & (close < open_)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 89)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_6h[i]) or np.isnan(ema89_12h_6h[i]) or 
            np.isnan(ema34_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend
        uptrend_12h = ema34_12h_6h[i] > ema89_12h_6h[i]
        downtrend_12h = ema34_12h_6h[i] < ema89_12h_6h[i]
        
        if position == 0:
            # Long: uptrend on 12h + price at 6h EMA34 + bullish rejection
            if (uptrend_12h and 
                close[i] <= ema34_6h[i] * 1.005 and  # near or slightly below EMA
                close[i] >= ema34_6h[i] * 0.995 and  # near or slightly above EMA
                is_bullish_rejection[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend on 12h + price at 6h EMA34 + bearish rejection
            elif (downtrend_12h and 
                  close[i] <= ema34_6h[i] * 1.005 and
                  close[i] >= ema34_6h[i] * 0.995 and
                  is_bearish_rejection[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or price moves significantly above EMA
            if not uptrend_12h or close[i] > ema34_6h[i] * 1.02:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or price moves significantly below EMA
            if not downtrend_12h or close[i] < ema34_6h[i] * 0.98:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA34_Trend_With_12h_Resistance_Support"
timeframe = "6h"
leverage = 1.0