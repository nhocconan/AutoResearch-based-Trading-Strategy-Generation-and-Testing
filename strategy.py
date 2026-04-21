#!/usr/bin/env python3
"""
4h_WickReversal_Volume_Spike
Hypothesis: Long wicks indicate rejection at key levels. A bullish reversal occurs when price makes a new low but closes in the upper 30% of the candle (long lower wick) with volume spike. Bearish reversal when price makes a new high but closes in the lower 30% (long upper wick) with volume spike. Uses 1d trend filter (price above/below 200 EMA) to avoid counter-trend trades. Works in bull/bear by only taking reversal trades aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(arr, period):
    """Calculate EMA with proper handling of NaN"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    ema = np.full_like(arr, np.nan, dtype=float)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        if np.isnan(ema[i-1]):
            ema[i] = np.mean(arr[i-period+1:i+1])
        else:
            ema[i] = (arr[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = calculate_ema(close_1d, 200)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike detection: volume > 1.5 * 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Wick calculations
    body_size = np.abs(close - open_)
    total_range = high - low
    lower_wick = np.where(close >= open_, open_ - low, close - low)
    upper_wick = np.where(close >= open_, high - close, high - open_)
    
    # Bullish reversal: long lower wick (>60% of range) + close in upper 30% + volume spike
    bullish_wick = (lower_wick > 0.6 * total_range) & (close > (low + 0.7 * total_range))
    bearish_wick = (upper_wick > 0.6 * total_range) & (close < (high - 0.3 * total_range))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(ema200_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema200 = ema200_aligned[i]
        vol_spike = volume_spike[i]
        bull_wick = bullish_wick[i]
        bear_wick = bearish_wick[i]
        
        if position == 0:
            # Long: bullish wick reversal + volume spike + price above 1d EMA200 (uptrend)
            if bull_wick and vol_spike and (price > ema200):
                signals[i] = 0.25
                position = 1
            # Short: bearish wick reversal + volume spike + price below 1d EMA200 (downtrend)
            elif bear_wick and vol_spike and (price < ema200):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below 1d EMA200 or opposite wick signal
            if price < ema200 or bear_wick:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above 1d EMA200 or opposite wick signal
            if price > ema200 or bull_wick:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WickReversal_Volume_Spike"
timeframe = "4h"
leverage = 1.0