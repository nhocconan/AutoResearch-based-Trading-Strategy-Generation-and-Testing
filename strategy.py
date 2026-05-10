#!/usr/bin/env python3
# 6H_1D_VolumeSpike_Breakout_Trend
# Hypothesis: Breakouts above 1d high or below 1d low with volume > 1.5x 1d volume average (20-period) confirm momentum in the direction of the 1d trend (close > EMA20 = bullish, close < EMA20 = bearish). Works in bull/bear by following 1d trend. Volume spike filters false breakouts. Target: 20-40 trades/year per symbol.

name = "6H_1D_VolumeSpike_Breakout_Trend"
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
    
    # Get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA20 for trend
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d 20-period volume average
    vol_series = pd.Series(volume_1d)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Breakout levels: today's high/low (using prior day's values for lookback-free)
    # We use the prior day's high/low to avoid look-ahead
    breakout_high = np.concatenate([[np.nan], high_1d[:-1]])
    breakout_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Trend: bullish if close > EMA20, bearish if close < EMA20
    bullish_trend = close_1d > ema20_1d
    bearish_trend = close_1d < ema20_1d
    
    # Volume spike: volume > 1.5x 20-period average
    vol_spike = volume_1d > (1.5 * vol_avg_20)
    
    # Align to 6h
    breakout_high_aligned = align_htf_to_ltf(prices, df_1d, breakout_high)
    breakout_low_aligned = align_htf_to_ltf(prices, df_1d, breakout_low)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_high_aligned[i]) or np.isnan(breakout_low_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        vol_spike = vol_spike_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + price breaks above prior day's high + volume spike
            if bullish and close[i] > breakout_high_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price breaks below prior day's low + volume spike
            elif bearish and close[i] < breakout_low_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price falls back below prior day's low
            if bearish or close[i] < breakout_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price rises back above prior day's high
            if bullish or close[i] > breakout_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals