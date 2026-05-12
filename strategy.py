#!/usr/bin/env python3
name = "6h_LiquidityVoid_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # 1w trend filter: EMA50 (to avoid counter-trend trades)
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Liquidity void detection: gap between candle low and prior candle high (or vice versa)
    # Bullish void: current candle low > previous candle high
    # Bearish void: current candle high < previous candle low
    bullish_void = (low > np.roll(high, 1)) & ~np.isnan(np.roll(high, 1))
    bearish_void = (high < np.roll(low, 1)) & ~np.isnan(np.roll(low, 1))
    
    # Volume filter: current volume > 1.5x average volume (to confirm institutional interest)
    volume_avg = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if 1w trend data not ready
        if np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish liquidity void + volume filter + 1w uptrend
            if bullish_void[i] and volume_filter[i] and (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish liquidity void + volume filter + 1w downtrend
            elif bearish_void[i] and volume_filter[i] and (close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when bearish void appears or price closes below entry area (simple: below prior candle low)
            if bearish_void[i] or (close[i] < np.roll(low, 1)[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when bullish void appears or price closes above entry area (simple: above prior candle high)
            if bullish_void[i] or (close[i] > np.roll(high, 1)[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals