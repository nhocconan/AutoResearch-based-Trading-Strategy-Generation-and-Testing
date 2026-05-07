#!/usr/bin/env python3
"""
1D_Engulfing_Pattern_1W_Trend_Filter
Hypothesis: Combine daily bullish/bearish engulfing candlestick patterns with weekly trend filter to capture high-probability reversals in trending markets.
Engulfing patterns signal strong momentum shifts; weekly trend ensures trades align with higher timeframe momentum.
Works in bull markets (bullish engulfing in uptrend) and bear markets (bearish engulfing in downtrend).
Targets 7-25 trades/year to minimize fee drag on daily timeframe.
"""
name = "1D_Engulfing_Pattern_1W_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data for pattern detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate bullish and bearish engulfing patterns
    bullish_engulfing = (close > open_price) & (open_price > np.roll(close, 1)) & (close > np.roll(open_price, 1))
    bearish_engulfing = (close < open_price) & (open_price < np.roll(close, 1)) & (close < np.roll(open_price, 1))
    
    # Align patterns to lower timeframe
    bullish_engulfing_aligned = align_htf_to_ltf(prices, df_1d, bullish_engulfing.astype(float))
    bearish_engulfing_aligned = align_htf_to_ltf(prices, df_1d, bearish_engulfing.astype(float))
    
    # Calculate 1W EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: current volume > 1.5 x 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(bullish_engulfing_aligned[i]) or np.isnan(bearish_engulfing_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing, weekly uptrend, and volume confirmation
            if (bullish_engulfing_aligned[i] == 1.0 and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing, weekly downtrend, and volume confirmation
            elif (bearish_engulfing_aligned[i] == 1.0 and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish engulfing or price closes below weekly EMA
            if (bearish_engulfing_aligned[i] == 1.0 or 
                close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish engulfing or price closes above weekly EMA
            if (bullish_engulfing_aligned[i] == 1.0 or 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals