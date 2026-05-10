#!/usr/bin/env python3
"""
6h_ABI_Trend_Follow
Hypothesis: Adaptive Bollinger Bands (BBands) with ATR-based width filtering
identify trend exhaustion and continuation phases. In strong trends (ADX>25),
price tends to ride the upper/lower BBand. Entries occur on pullbacks to the
20-period EMA within the trend, with exits when price closes outside the
Adaptive BBands. Uses 12h trend filter (EMA50) to ensure alignment with higher
timeframe momentum. Works in bull/bear by following 12h trend direction.
Target: 20-40 trades/year per symbol.
"""

name = "6h_ABI_Trend_Follow"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend_12h = close_12h > ema50_12h
    bearish_trend_12h = close_12h < ema50_12h
    
    # Adaptive Bollinger Bands (6h)
    # Base: 20-period SMA, width: ATR(14) * 2
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    bb_width = atr14 * 2
    upper_band = sma20 + bb_width
    lower_band = sma20 - bb_width
    
    # 6h EMA20 for pullback entries
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h trend to 6h
    bullish_aligned = align_htf_to_ltf(prices, df_12h, bullish_trend_12h.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_12h, bearish_trend_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema20[i]) or np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish 12h trend + price pulls back to EMA20 and closes above it
            if bullish and close[i] > ema20[i] and close[i-1] <= ema20[i-1]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish 12h trend + price pulls back to EMA20 and closes below it
            elif bearish and close[i] < ema20[i] and close[i-1] >= ema20[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below lower adaptive Bollinger Band
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above upper adaptive Bollinger Band
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals