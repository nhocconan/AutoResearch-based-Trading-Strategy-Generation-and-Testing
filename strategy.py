#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian_Breakout_VolumeTrend_1dFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Donchian Channel (20-period) on 4h
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # ATR (20-period) for volatility filter
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.mean(tr[i-19:i+1])
    
    # Volume MA (20-period) for volume confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    
    # Daily trend filter: EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: breakout above upper band + volume + uptrend
            if (close[i] > highest_high[i] and 
                volume[i] > vol_ma[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band + volume + downtrend
            elif (close[i] < lowest_low[i] and 
                  volume[i] > vol_ma[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        # Exit conditions
        elif position == 1:
            # Exit long: close below midpoint OR ATR-based stop
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint or close[i] < prices['high'][:i+1].max() - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above midpoint OR ATR-based stop
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint or close[i] > prices['low'][:i+1].min() + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian breakouts capture trends, volume confirms conviction,
# and daily EMA50 filter ensures alignment with higher-timeframe trend.
# Works in bull markets (captures uptrends) and bear markets (captures downtrends).
# 4h timeframe balances responsiveness with low frequency to minimize fee drag.
# Target: 75-200 trades over 4 years (19-50/year) to stay within profitable range.