#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Bollinger Band breakout with 1w trend filter
    # Long when: price breaks above BB(20,2) upper band AND 1w EMA200 is rising (bull trend)
    # Short when: price breaks below BB(20,2) lower band AND 1w EMA200 is falling (bear trend)
    # Exit when: price crosses BB middle band (20 SMA) or 1w EMA200 trend reverses
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via 1w EMA200 trend filter preventing counter-trend trades.
    # Bollinger Bands provide volatility-adjusted breakout levels.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Bollinger Bands (20,2) on 6h
    lookback = 20
    sma20 = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().values
    std20 = pd.Series(close).rolling(window=lookback, min_periods=lookback).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_middle = sma20  # 20 SMA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Band conditions
        price_above_upper = close[i] > bb_upper[i]
        price_below_lower = close[i] < bb_lower[i]
        price_cross_above_middle = close[i] > bb_middle[i] and close[i-1] <= bb_middle[i-1]
        price_cross_below_middle = close[i] < bb_middle[i] and close[i-1] >= bb_middle[i-1]
        
        # 1w EMA200 trend filter (rising/falling)
        if i >= 101:  # Need previous bar for trend check
            ema200_rising = ema200_1w_aligned[i] > ema200_1w_aligned[i-1]
            ema200_falling = ema200_1w_aligned[i] < ema200_1w_aligned[i-1]
        else:
            ema200_rising = False
            ema200_falling = False
        
        # Entry conditions
        long_entry = price_above_upper and ema200_rising and position != 1
        short_entry = price_below_lower and ema200_falling and position != -1
        
        # Exit conditions
        exit_long = price_cross_below_middle or (position == 1 and not ema200_rising)
        exit_short = price_cross_above_middle or (position == -1 and not ema200_falling)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_bb_breakout_trend_filter_v1"
timeframe = "6h"
leverage = 1.0