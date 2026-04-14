#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    # Daily EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channel (20) on 12h
    lookback = 20
    # Use pandas rolling on series
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=lookback, min_periods=lookback).max().values
    donch_low = low_series.rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average (20)
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25%
    
    start = max(lookback, 50*2, 20)  # ensure enough data
    start = 120  # conservative
    
    for i in range(start, n):
        # skip if any needed value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Determine bias from daily EMA50
            bullish_bias = price > ema50_1d_aligned[i]
            bearish_bias = price < ema50_1d_aligned[i]
            # Long breakout with volume confirmation
            if bullish_bias and price > donch_high[i] and vol > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short breakout
            elif bearish_bias and price < donch_low[i] and vol > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price re-enters channel (below upper band) OR bias flips?
            if price < donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short when price > lower band
            if price > donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_DailyEMA50_Volume"
timeframe = "12h"
leverage = 1.0